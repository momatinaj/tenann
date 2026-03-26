/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "tenann/searcher/acorn_hnsw_ann_searcher.h"

#include <algorithm>
#include <queue>
#include <unordered_set>
#include <vector>

#include "faiss/IndexHNSW.h"
#include "faiss/IndexIDMap.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/impl/FaissException.h"
#include "faiss/impl/HNSW.h"
#include "tenann/common/logging.h"
#include "tenann/index/internal/faiss_index_util.h"
#include "tenann/index/parameter_serde.h"
#include "tenann/searcher/internal/id_filter_adapter.h"
#include "tenann/store/index_meta.h"
#include "tenann/util/distance_util.h"

namespace tenann {

namespace acorn_detail {
using namespace faiss;
using storage_idx_t = HNSW::storage_idx_t;

DistanceComputer* storage_distance_computer(const faiss::Index* storage) {
  if (storage->metric_type == METRIC_INNER_PRODUCT) {
    T_LOG(ERROR) << "inner product is not supported for ACORN search";
    return nullptr;
  }
  return storage->get_distance_computer();
}

void greedy_update_nearest(const HNSW& hnsw, DistanceComputer& qdis, int level,
                           storage_idx_t& nearest, float& d_nearest) {
  for (;;) {
    storage_idx_t prev_nearest = nearest;
    size_t begin, end;
    hnsw.neighbor_range(nearest, level, &begin, &end);
    for (size_t i = begin; i < end; i++) {
      storage_idx_t v = hnsw.neighbors[i];
      if (v < 0) break;
      float dis = qdis(v);
      if (dis < d_nearest) {
        nearest = v;
        d_nearest = dis;
      }
    }
    if (nearest == prev_nearest) return;
  }
}

/// Collect 1-hop neighbors of node `v` at `level` from the HNSW graph.
inline void get_neighbors(const HNSW& hnsw, storage_idx_t v, int level,
                          std::vector<storage_idx_t>& out) {
  out.clear();
  size_t begin, end;
  hnsw.neighbor_range(v, level, &begin, &end);
  for (size_t i = begin; i < end; i++) {
    storage_idx_t n = hnsw.neighbors[i];
    if (n < 0) break;
    out.push_back(n);
  }
}

/// ACORN-1 GET-NEIGHBORS: expand to 2-hop, filter by predicate, truncate to M.
///
/// For visited node `v`:
///   1. Collect N(v) = 1-hop neighbors
///   2. For each n in N(v), collect N(n) = neighbors of n
///   3. expanded = N(v) UNION all N(n), deduplicated, excluding already-visited
///   4. Filter: keep only nodes where sel->is_member(id) is true
///   5. Compute distances, sort by distance, take first M
///   6. Return as effective neighborhood
void acorn_get_neighbors(const HNSW& hnsw, DistanceComputer& qdis, storage_idx_t v,
                         int level, int M, const IDSelector* sel, VisitedTable& vt,
                         std::vector<std::pair<float, storage_idx_t>>& result) {
  result.clear();

  std::vector<storage_idx_t> one_hop;
  get_neighbors(hnsw, v, level, one_hop);

  // Collect unique 2-hop expanded set
  std::unordered_set<storage_idx_t> expanded_set;
  for (auto n : one_hop) {
    expanded_set.insert(n);
  }

  std::vector<storage_idx_t> two_hop;
  for (auto n : one_hop) {
    get_neighbors(hnsw, n, level, two_hop);
    for (auto nn : two_hop) {
      expanded_set.insert(nn);
    }
  }

  // Remove `v` itself from the expanded set
  expanded_set.erase(v);

  // Filter by predicate and compute distances
  for (auto candidate : expanded_set) {
    if (vt.get(candidate)) continue;
    if (sel && !sel->is_member(candidate)) continue;
    float d = qdis(candidate);
    result.emplace_back(d, candidate);
  }

  // Sort by distance ascending and truncate to M
  std::sort(result.begin(), result.end());
  if (static_cast<int>(result.size()) > M) {
    result.resize(M);
  }
}

/// ACORN-1 search on level 0: predicate subgraph traversal with 2-hop expansion.
///
/// Implements Algorithm 2 from the ACORN paper adapted for ACORN-1:
///   - Uses a dynamic candidate list (MinimaxHeap of size efSearch)
///   - At each visited node, calls acorn_get_neighbors for expanded+filtered neighborhood
///   - Maintains a result set W of ef nearest predicate-passing neighbors
void AcornSearchFromCandidates(const HNSW& hnsw, DistanceComputer& qdis,
                               int k, int ef, const IDSelector* sel,
                               storage_idx_t entry, float d_entry,
                               std::vector<idx_t>& result_ids,
                               std::vector<float>& result_distances,
                               VisitedTable& vt) {
  int M = hnsw.nb_neighbors(0) / 2;
  if (M <= 0) M = 16;

  // W: result heap (max-heap by distance, so we can pop farthest)
  using Node = std::pair<float, storage_idx_t>;
  std::priority_queue<Node> W;

  // C: candidate min-heap (pop nearest first)
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> C;

  vt.set(entry);
  float d = d_entry;

  if (!sel || sel->is_member(entry)) {
    W.push({d, entry});
  }
  C.push({d, entry});

  std::vector<std::pair<float, storage_idx_t>> neighborhood;

  while (!C.empty()) {
    auto [d_c, c] = C.top();
    C.pop();

    // Early termination: if the nearest candidate is farther than the farthest
    // result and we already have enough results
    if (static_cast<int>(W.size()) >= ef && d_c > W.top().first) {
      break;
    }

    acorn_get_neighbors(hnsw, qdis, c, 0, M, sel, vt, neighborhood);

    for (auto& [d_n, n] : neighborhood) {
      if (vt.get(n)) continue;
      vt.set(n);

      bool should_add = (static_cast<int>(W.size()) < ef);
      if (!should_add && !W.empty() && d_n < W.top().first) {
        should_add = true;
      }

      if (should_add) {
        C.push({d_n, n});
        if (!sel || sel->is_member(n)) {
          W.push({d_n, n});
          if (static_cast<int>(W.size()) > ef) {
            W.pop();
          }
        }
      }
    }
  }

  // Extract top-k from W (W is a max-heap, so reverse)
  int n_results = std::min(static_cast<int>(W.size()), static_cast<int>(k));
  result_ids.resize(n_results);
  result_distances.resize(n_results);

  for (int i = n_results - 1; i >= 0; i--) {
    auto [dist, id] = W.top();
    W.pop();
    result_ids[i] = id;
    result_distances[i] = dist;
  }
}

/// Full ACORN-1 search: greedy descent on upper levels, then ACORN search on level 0.
void AcornSearch(const IndexHNSW& index, const float* x, int64_t k,
                 int64_t* I, float* D, const SearchParametersHNSW* params,
                 const IDSelector* sel) {
  if (index.hnsw.entry_point == -1) {
    for (int64_t i = 0; i < k; i++) {
      I[i] = -1;
      D[i] = std::numeric_limits<float>::max();
    }
    return;
  }

  int efSearch = params ? params->efSearch : index.hnsw.efSearch;
  int ef = std::max(efSearch, static_cast<int>(k));

  DistanceComputer* p_dis = storage_distance_computer(index.storage);
  std::unique_ptr<DistanceComputer> del(p_dis);
  auto& dis = *p_dis;
  dis.set_query(x);

  // Phase 1: Greedy descent on upper levels (same as standard HNSW)
  storage_idx_t nearest = index.hnsw.entry_point;
  float d_nearest = dis(nearest);

  for (int level = index.hnsw.max_level; level >= 1; level--) {
    greedy_update_nearest(index.hnsw, dis, level, nearest, d_nearest);
  }

  // Phase 2: ACORN-1 search on level 0
  VisitedTable vt(index.ntotal);
  std::vector<idx_t> res_ids;
  std::vector<float> res_dists;

  AcornSearchFromCandidates(index.hnsw, dis, k, ef, sel, nearest, d_nearest,
                            res_ids, res_dists, vt);

  for (int64_t i = 0; i < k; i++) {
    if (i < static_cast<int64_t>(res_ids.size())) {
      I[i] = res_ids[i];
      D[i] = res_dists[i];
    } else {
      I[i] = -1;
      D[i] = std::numeric_limits<float>::max();
    }
  }
}

}  // namespace acorn_detail

AcornHnswAnnSearcher::AcornHnswAnnSearcher(const IndexMeta& meta) : AnnSearcher(meta) {
  FetchParameters(meta, &search_params_);
}

AcornHnswAnnSearcher::~AcornHnswAnnSearcher() = default;

void AcornHnswAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int64_t k,
                                     int64_t* result_id, const IdFilter* id_filter) {
  std::vector<float> distances(k);
  AnnSearch(query_vector, k, result_id, reinterpret_cast<uint8_t*>(distances.data()), id_filter);
}

void AcornHnswAnnSearcher::AnnSearch(PrimitiveSeqView query_vector, int64_t k,
                                     int64_t* result_ids, uint8_t* result_distances,
                                     const IdFilter* id_filter) {
  try {
    T_CHECK_NOTNULL(index_ref_);
    T_CHECK_EQ(index_ref_->index_type(), IndexType::kFaissHnsw);
    T_CHECK_EQ(query_vector.elem_type, PrimitiveType::kFloatType);

    std::shared_ptr<IdFilterAdapter> id_filter_adapter;
    const faiss::IDSelector* faiss_sel = nullptr;

    if (id_filter) {
      if (faiss_id_map_ != nullptr) {
        id_filter_adapter = IdFilterAdapterFactory::CreateIdFilterAdapter(
            id_filter, &reinterpret_cast<const faiss::IndexIDMap*>(faiss_id_map_)->id_map);
      } else {
        id_filter_adapter = IdFilterAdapterFactory::CreateIdFilterAdapter(id_filter);
      }
      faiss_sel = id_filter_adapter.get();
    }

    const float* x = reinterpret_cast<const float*>(query_vector.data);
    const faiss::IndexHNSW* hnsw_index =
        reinterpret_cast<const faiss::IndexHNSW*>(faiss_hnsw_);

    faiss::SearchParametersHNSW faiss_params;
    faiss_params.efSearch = search_params_.efSearch;
    faiss_params.check_relative_distance = search_params_.check_relative_distance;

    if (faiss_transform_ != nullptr) {
      const float* xt = reinterpret_cast<const faiss::IndexPreTransform*>(faiss_transform_)
                            ->apply_chain(ANN_SEARCHER_QUERY_COUNT, x);
      std::unique_ptr<const float[]> del(xt == x ? nullptr : xt);
      acorn_detail::AcornSearch(*hnsw_index, xt, k,
                                result_ids, reinterpret_cast<float*>(result_distances),
                                &faiss_params, faiss_sel);
    } else {
      acorn_detail::AcornSearch(*hnsw_index, x, k,
                                result_ids, reinterpret_cast<float*>(result_distances),
                                &faiss_params, faiss_sel);
    }

    // Map internal IDs back to external IDs if using IndexIDMap
    if (faiss_id_map_ != nullptr) {
      for (int64_t i = 0; i < k; i++) {
        if (result_ids[i] >= 0) {
          result_ids[i] =
              reinterpret_cast<const faiss::IndexIDMap*>(faiss_id_map_)->id_map[result_ids[i]];
        }
      }
    }

    if (common_params_.metric_type == MetricType::kCosineSimilarity) {
      auto distances = reinterpret_cast<float*>(result_distances);
      L2DistanceToCosineSimilarity(distances, distances, k);
    }
  }
  CATCH_FAISS_ERROR
}

void AcornHnswAnnSearcher::RangeSearch(PrimitiveSeqView query_vector, float range, int64_t limit,
                                       ResultOrder result_order, std::vector<int64_t>* result_ids,
                                       std::vector<float>* result_distances,
                                       const IdFilter* id_filter) {
  T_LOG(WARNING) << "ACORN-1 range search is not implemented; falling back to standard HNSW range "
                    "search behavior via top-K with distance threshold.";

  // For range search, delegate to AnnSearch with a large K and post-filter by distance
  int64_t large_k = limit > 0 ? limit : 1000;
  std::vector<int64_t> ids(large_k);
  std::vector<float> dists(large_k);

  AnnSearch(query_vector, large_k, ids.data(), reinterpret_cast<uint8_t*>(dists.data()), id_filter);

  result_ids->clear();
  result_distances->clear();

  float radius = range;
  if (common_params_.metric_type == MetricType::kCosineSimilarity) {
    radius = CosineSimilarityThresholdToL2Distance(range);
  }

  for (int64_t i = 0; i < large_k; i++) {
    if (ids[i] < 0) break;
    if (dists[i] <= radius) {
      result_ids->push_back(ids[i]);
      result_distances->push_back(dists[i]);
    }
  }
}

void AcornHnswAnnSearcher::OnSearchParamItemChange(const std::string& key, const json& value) {
  try {
    if (key == FaissHnswSearchParams::efSearch_key) {
      value.is_number_integer() && (search_params_.efSearch = value.get<int>());
    } else if (key == FaissHnswSearchParams::check_relative_distance_key) {
      value.is_boolean() && (search_params_.check_relative_distance = value.get<bool>());
    } else {
      T_LOG(WARNING) << "Unsupported search parameter: " << key;
    }
  }
  CATCH_JSON_ERROR
}

void AcornHnswAnnSearcher::OnSearchParamsChange(const json& value) {
  for (auto it = value.begin(); it != value.end(); ++it) {
    OnSearchParamItemChange(it.key(), it.value());
  }
}

void AcornHnswAnnSearcher::OnIndexLoaded() {
  auto faiss_index = static_cast<faiss::Index*>(index_ref_->index_raw());
  auto [id_map, transform, hnsw] = faiss_util::CheckAndUnpackHnsw(faiss_index, &common_params_);
  faiss_id_map_ = id_map;
  faiss_transform_ = transform;
  faiss_hnsw_ = hnsw;
}

}  // namespace tenann
