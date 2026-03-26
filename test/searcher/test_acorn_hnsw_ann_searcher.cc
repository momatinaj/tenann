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

#include <cmath>
#include <iostream>
#include <random>
#include <unordered_set>

#include "tenann/index/parameters.h"
#include "tenann/searcher/acorn_hnsw_ann_searcher.h"
#include "test/faiss_test_base.h"

namespace tenann {

class AcornHnswAnnSearcherTest : public FaissTestBase {
 public:
  AcornHnswAnnSearcherTest() : FaissTestBase() {
    InitFaissHnswMeta();
    faiss_hnsw_index_builder_ = IndexFactory::CreateBuilderFromMeta(faiss_hnsw_meta_);
  }

  IndexMeta MakeAcornMeta() {
    IndexMeta acorn_meta = faiss_hnsw_meta_;
    acorn_meta.SetIndexType(IndexType::kFaissHnswAcorn);
    return acorn_meta;
  }
};

TEST_F(AcornHnswAnnSearcherTest, BasicSearch_NoFilter) {
  CreateAndWriteFaissHnswIndex(true);

  auto acorn_meta = MakeAcornMeta();
  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(acorn_meta);
  searcher->ReadIndex(index_with_primary_key_path());

  result_ids_.clear();
  result_ids_.resize(nq_ * k_);
  for (size_t i = 0; i < nq_; i++) {
    searcher->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
  }

  float recall = ComputeRecall();
  std::cout << "[ACORN-1 NoFilter] Recall: " << recall << std::endl;
  EXPECT_GT(recall, 0.5) << "ACORN-1 without filter should achieve reasonable recall";
}

TEST_F(AcornHnswAnnSearcherTest, BasicSearch_WithDistances) {
  CreateAndWriteFaissHnswIndex(true);

  auto acorn_meta = MakeAcornMeta();
  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(acorn_meta);
  searcher->ReadIndex(index_with_primary_key_path());

  std::vector<float> distances(k_);
  result_ids_.resize(k_);

  searcher->AnnSearch(query_view_[0], k_, result_ids_.data(),
                      reinterpret_cast<uint8_t*>(distances.data()));

  bool distances_sorted = true;
  for (uint32_t i = 1; i < k_; i++) {
    if (distances[i] < distances[i - 1]) {
      distances_sorted = false;
      break;
    }
  }
  EXPECT_TRUE(distances_sorted) << "Results should be sorted by ascending distance";

  for (uint32_t i = 0; i < k_; i++) {
    EXPECT_GE(distances[i], 0.0f) << "L2 distances must be non-negative";
  }
}

TEST_F(AcornHnswAnnSearcherTest, FilteredSearch_AllReject) {
  CreateAndWriteFaissHnswIndex(true);

  auto acorn_meta = MakeAcornMeta();
  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(acorn_meta);
  searcher->ReadIndex(index_with_primary_key_path());

  class RejectAllFilter : public IdFilter {
   public:
    bool IsMember(idx_t) const override { return false; }
    ~RejectAllFilter() override = default;
  } reject_filter;

  result_ids_.resize(k_);
  std::fill(result_ids_.begin(), result_ids_.end(), 0);
  searcher->AnnSearch(query_view_[0], k_, result_ids_.data(), &reject_filter);

  EXPECT_TRUE(std::all_of(result_ids_.begin(), result_ids_.end(),
                           [](int64_t id) { return id == -1; }))
      << "With all-reject filter, all result IDs should be -1";
}

TEST_F(AcornHnswAnnSearcherTest, FilteredSearch_PartialAccept) {
  CreateAndWriteFaissHnswIndex(true, id_filter_count_);

  auto acorn_meta = MakeAcornMeta();
  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(acorn_meta);
  searcher->ReadIndex(index_with_primary_key_path());

  RangeIdFilter range_filter(0, id_filter_count_, false);

  result_ids_.resize(nq_ * k_);
  for (size_t i = 0; i < nq_; i++) {
    searcher->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_, &range_filter);
  }

  float recall = ComputeRecall();
  std::cout << "[ACORN-1 RangeFilter] Recall: " << recall << std::endl;
  EXPECT_GT(recall, 0.3) << "ACORN-1 with range filter should achieve non-trivial recall";
}

TEST_F(AcornHnswAnnSearcherTest, FilteredSearch_BitmapFilter) {
  CreateAndWriteFaissHnswIndex(true, id_filter_count_);

  auto acorn_meta = MakeAcornMeta();
  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(acorn_meta);
  searcher->ReadIndex(index_with_primary_key_path());

  std::vector<uint8_t> bitmap((nb_ + 7) / 8, 0);
  for (int i = 0; i < id_filter_count_ && i < static_cast<int>(nb_); ++i) {
    uint64_t id = ids_[i];
    bitmap[id >> 3] |= (1 << (id & 7));
  }
  BitmapIdFilter bitmap_filter(bitmap.data(), bitmap.size());

  result_ids_.resize(nq_ * k_);
  for (size_t i = 0; i < nq_; i++) {
    searcher->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_, &bitmap_filter);
  }

  float recall = ComputeRecall();
  std::cout << "[ACORN-1 BitmapFilter] Recall: " << recall << std::endl;
  EXPECT_GT(recall, 0.3) << "ACORN-1 with bitmap filter should achieve non-trivial recall";
}

TEST_F(AcornHnswAnnSearcherTest, SameIndexDifferentSearcher) {
  CreateAndWriteFaissHnswIndex(true);

  // Standard HNSW search
  auto hnsw_searcher = AnnSearcherFactory::CreateSearcherFromMeta(faiss_hnsw_meta());
  hnsw_searcher->ReadIndex(index_with_primary_key_path());

  std::vector<int64_t> hnsw_results(nq_ * k_);
  for (size_t i = 0; i < nq_; i++) {
    hnsw_searcher->AnnSearch(query_view_[i], k_, hnsw_results.data() + i * k_);
  }

  // ACORN-1 search over the same index
  auto acorn_meta = MakeAcornMeta();
  auto acorn_searcher = AnnSearcherFactory::CreateSearcherFromMeta(acorn_meta);
  acorn_searcher->ReadIndex(index_with_primary_key_path());

  std::vector<int64_t> acorn_results(nq_ * k_);
  for (size_t i = 0; i < nq_; i++) {
    acorn_searcher->AnnSearch(query_view_[i], k_, acorn_results.data() + i * k_);
  }

  // Both should return valid results (not all -1)
  bool hnsw_has_valid = std::any_of(hnsw_results.begin(), hnsw_results.end(),
                                     [](int64_t id) { return id >= 0; });
  bool acorn_has_valid = std::any_of(acorn_results.begin(), acorn_results.end(),
                                      [](int64_t id) { return id >= 0; });

  EXPECT_TRUE(hnsw_has_valid) << "Standard HNSW should return valid results";
  EXPECT_TRUE(acorn_has_valid) << "ACORN-1 should return valid results on the same index";
}

TEST_F(AcornHnswAnnSearcherTest, AcornSearch_WithoutIDMap) {
  CreateAndWriteFaissHnswIndex(false);

  auto acorn_meta = MakeAcornMeta();
  auto searcher = AnnSearcherFactory::CreateSearcherFromMeta(acorn_meta);
  searcher->ReadIndex(index_path());

  result_ids_.resize(nq_ * k_);
  for (size_t i = 0; i < nq_; i++) {
    searcher->AnnSearch(query_view_[i], k_, result_ids_.data() + i * k_);
  }

  float recall = ComputeRecall();
  std::cout << "[ACORN-1 NoIDMap] Recall: " << recall << std::endl;
  EXPECT_GT(recall, 0.5) << "ACORN-1 without IDMap should work correctly";
}

}  // namespace tenann
