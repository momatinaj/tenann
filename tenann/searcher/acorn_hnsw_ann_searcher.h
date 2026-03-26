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

#pragma once

#include "tenann/searcher/ann_searcher.h"

namespace tenann {

/// ACORN-1 hybrid search over a standard FAISS HNSW index.
///
/// Uses the same index file as FaissHnswAnnSearcher but modifies the search
/// algorithm to implement predicate subgraph traversal with 2-hop neighbor
/// expansion, as described in:
///   Patel et al., "ACORN: Performant and Predicate-Agnostic Search Over
///   Vector Embeddings and Structured Data", SIGMOD 2024.
///
/// ACORN-1 key idea: at each visited node during greedy search on level 0,
/// expand the neighbor list to include all 1-hop AND 2-hop neighbors, apply
/// a predicate filter, truncate to M, and use that as the effective
/// neighborhood. This enables efficient traversal of the "predicate subgraph"
/// without requiring index construction changes.
class AcornHnswAnnSearcher : public AnnSearcher {
 public:
  explicit AcornHnswAnnSearcher(const IndexMeta& meta);
  ~AcornHnswAnnSearcher() override;

  T_FORBID_MOVE(AcornHnswAnnSearcher);
  T_FORBID_COPY_AND_ASSIGN(AcornHnswAnnSearcher);

  void AnnSearch(PrimitiveSeqView query_vector, int64_t k, int64_t* result_id,
                 const IdFilter* id_filter = nullptr) override;

  void AnnSearch(PrimitiveSeqView query_vector, int64_t k, int64_t* result_ids,
                 uint8_t* result_distances, const IdFilter* id_filter = nullptr) override;

  void RangeSearch(PrimitiveSeqView query_vector, float range, int64_t limit,
                   ResultOrder result_order, std::vector<int64_t>* result_ids,
                   std::vector<float>* result_distances,
                   const IdFilter* id_filter = nullptr) override;

 protected:
  void OnSearchParamItemChange(const std::string& key, const json& value) override;
  void OnSearchParamsChange(const json& value) override;
  void OnIndexLoaded() override;

 private:
  FaissHnswSearchParams search_params_;
  const void* faiss_id_map_;
  const void* faiss_transform_;
  const void* faiss_hnsw_;
};

}  // namespace tenann
