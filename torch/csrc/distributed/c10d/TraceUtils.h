#pragma once
#include <c10/core/ScalarType.h>
#include <c10/util/ApproximateClock.h>
#include <c10/util/irange.h>
<<<<<<< HEAD
#include <c10/util/string_view.h>
=======
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/profiler/combined_traceback.h>

<<<<<<< HEAD
#include <sys/types.h>
#include <cstdlib>
=======
#include <fmt/compile.h>
#include <fmt/core.h>
#include <fmt/ostream.h> // optional, for ostream fallback
#include <fmt/ranges.h> // for fmt::join

#include <sys/types.h>
#include <cstdlib>
#include <cstring>
#include <iterator>
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
#include <string>
#include <vector>

namespace c10d {

<<<<<<< HEAD
// A struct to hold the latest status of the process group.
struct ProcessGroupStatus {
  // the sequential number of the last collective enqueued into workMetaList_
  // This is useful for indentifying a rank that has not join a collective
  // initialized to be -1 to indicate no collective has been enqueued
  int64_t lastEnqueuedSeq{-1};
  // the sequential number of the last collective started as the kernel
  int64_t lastStartedSeq{-1};
  // the sequential number of the last collective completed marked by
  // the watchdog thread
  // initialized to be -1 to indicate no collective has been completed
  int64_t lastCompletedSeq{-1};

  // the name of the last collective enqueued into workMetaList_
  std::string lastEnqueuedWorkName;
  // the name of the last collective started as the kernel
  std::string lastStartedWorkName;
  // the name of the last collective completed
  std::string lastCompletedWorkName;

  // the sizes of the last work enqueued
  size_t lastEnqueuedNumelIn;
  size_t lastEnqueuedNumelOut;
  // the sizes of the last work completed
  size_t lastCompletedNumelIn;
  size_t lastCompletedNumelOut;
  // the sizes of the last work started
  size_t lastStartedNumelIn;
  size_t lastStartedNumelOut;
};

inline std::string getTraceStartKey(const std::string& pgName, int rank) {
  return pgName + "_" + std::to_string(rank) + "_trace_start";
}

inline std::string getTraceEndKey(const std::string& pgName, int rank) {
  return pgName + "_" + std::to_string(rank) + "_trace_end";
=======
inline std::string getTraceStartKey(const std::string& pgName, int rank) {
  return fmt::format(FMT_COMPILE("{}_{}_trace_start"), pgName, rank);
}

inline std::string getTraceEndKey(const std::string& pgName, int rank) {
  return fmt::format(FMT_COMPILE("{}_{}_trace_end"), pgName, rank);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

inline bool traceUpdate(
    c10::intrusive_ptr<Store>& store,
    const std::string& key,
    uint64_t seq,
    const std::string& col) {
  std::vector<uint8_t> value(col.size() + sizeof(seq) + 1);
<<<<<<< HEAD
  memcpy(value.data(), &seq, sizeof(seq));
  memcpy(value.data() + sizeof(seq), col.data(), col.size());
=======
  std::memcpy(value.data(), &seq, sizeof(seq));
  std::memcpy(value.data() + sizeof(seq), col.data(), col.size());
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
  try {
    store->set(key, value);
    return true;
  } catch (...) {
    LOG(ERROR) << "Store is down while updating #" << seq << " with key "
               << key;
    return false;
  }
  return true;
}

enum TraceDebugEvent {
  kEventStart,
  kEventEnd,
};
// <seq, <rank, <col, start/end>>>
using TraceMap =
    std::map<uint64_t, std::map<int, std::pair<std::string, TraceDebugEvent>>>;

inline std::string ranksToString(const std::vector<int>& ranks) {
<<<<<<< HEAD
  std::string str;
  for (int rank : ranks) {
    if (str.empty()) {
      str = std::to_string(rank);
    } else {
      str += ", " + std::to_string(rank);
    }
  }
  return str;
=======
  return fmt::to_string(fmt::join(ranks, ", "));
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

inline std::string ranksFromTrace(
    const std::vector<std::pair<int, std::string>>& items) {
<<<<<<< HEAD
  std::string ranks;
  for (auto& p : items) {
    if (ranks.empty()) {
      ranks = std::to_string(p.first);
    } else {
      ranks += ", " + std::to_string(p.first);
    }
  }
  return ranks;
=======
  fmt::memory_buffer buf;
  bool first = true;
  for (const auto& [rank, _] : items) {
    if (!first) {
      fmt::format_to(std::back_inserter(buf), ", ");
    }
    fmt::format_to(std::back_inserter(buf), "{}", rank);
    first = false;
  }
  return fmt::to_string(buf);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

inline std::string analyzeMissingRanks(const std::vector<int>& missingRanks) {
  return c10::str(
      "\n\t - To our best knowledge, ranks [",
      ranksToString(missingRanks),
      "] are the lagging ranks that caused this timeout. "
      "They never joined any collectives");
}

inline std::string analyzeLaggingRanks(const TraceMap& traceMap) {
  uint64_t lagSeq = traceMap.begin()->first;
  std::vector<int> startRanks;
  std::vector<int> endRanks;
  for (auto& p : traceMap.begin()->second) {
    if (p.second.second == kEventStart) {
      startRanks.push_back(p.first);
    } else {
      endRanks.push_back(p.first);
    }
  }
  std::string report =
      "\n\t - To our best knowledge, the lagging/dead/mismatched ranks "
      "that caused the desync are:";
  if (!startRanks.empty()) {
    report += c10::str(
        "\n\t   - [",
        ranksToString(startRanks),
        "] joined but didn't finish collective #",
        lagSeq,
        " (count from 1)");
  }
  if (!endRanks.empty()) {
    report += c10::str(
        "\n\t     [",
        ranksToString(endRanks),
        "] finished collective #",
        lagSeq,
        ", but didn't join collective #",
        lagSeq + 1,
        " (count from 1)");
  }
  return report;
}

inline std::string dumpSnapshot(TraceMap& traceMap) {
  std::string report = "\n\t - Snapshot of ranks' latest states:";
  for (auto& tracePair : traceMap) {
    uint64_t seq = tracePair.first;
    std::map<int, std::pair<std::string, TraceDebugEvent>>& subMap =
        tracePair.second;

    std::unordered_map<std::string, std::vector<int>> collectivesStart;
    std::unordered_map<std::string, std::vector<int>> collectivesEnd;
<<<<<<< HEAD
    for (auto& p : subMap) {
=======
    for (const auto& p : subMap) {
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
      int rank = p.first;
      const std::string& col = p.second.first;
      if (p.second.second == kEventStart) {
        collectivesStart[col].push_back(rank);
      } else {
        collectivesEnd[col].push_back(rank);
      }
    }

    if (!collectivesStart.empty()) {
      report += c10::str("\n\t   #", seq, " started ranks:");
      for (auto& mapPair : collectivesStart) {
        report += c10::str(
            "\n\t     [",
            ranksToString(mapPair.second),
            "] started ",
            mapPair.first);
      }
    }
    if (!collectivesEnd.empty()) {
      report += c10::str("\n\t   #", seq, " finished ranks:");
      for (auto& mapPair : collectivesEnd) {
        report += c10::str(
            "\n\t     [",
            ranksToString(mapPair.second),
            "] finished ",
            mapPair.first);
      }
    }
  }
  return report;
}

inline bool parseTraceValue(
    c10::intrusive_ptr<Store>& store,
    const std::string& key,
    uint64_t& seq,
    std::string& col) {
  try {
    std::vector<uint8_t> traceValue = store->get(key);
<<<<<<< HEAD
    memcpy(&seq, traceValue.data(), sizeof(seq));
=======
    std::memcpy(&seq, traceValue.data(), sizeof(seq));
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
    std::string colName((char*)traceValue.data() + sizeof(seq));
    col = colName;
    return true;
  } catch (...) {
    LOG(ERROR) << "Store is down while getting key " << key;
    return false;
  }
  return true;
}

inline std::string retrieveDesyncReport(
    c10::intrusive_ptr<Store>& store,
    const std::string& pgName,
    int myRank,
    int worldSize) {
  std::string report;

  uint64_t thisSeq = 0;
  std::string thisCol;

  std::vector<int> missingRanks;
  TraceMap traceMap;

  for (const auto rank : c10::irange(worldSize)) {
    // Build traceMapStart.
    uint64_t seqStart = 0;
    {
      std::string traceKeyStart = getTraceStartKey(pgName, rank);
      if (!store->check({traceKeyStart})) {
        missingRanks.push_back(rank);
        continue;
      }
      std::string col;
      if (!parseTraceValue(store, traceKeyStart, seqStart, col)) {
        return report;
      }
      traceMap[seqStart].emplace(rank, std::make_pair(col, kEventStart));
      if (rank == myRank) {
        thisSeq = seqStart;
        thisCol = std::move(col);
      }
    }

    // Build traceMapEnd.
    {
      std::string traceKeyEnd = getTraceEndKey(pgName, rank);
      if (!store->check({traceKeyEnd})) {
        continue;
      }
      uint64_t seq = 0;
      std::string col;
      if (!parseTraceValue(store, traceKeyEnd, seq, col)) {
        return report;
      }
      if (seq == seqStart) {
        traceMap[seq][rank].second = kEventEnd;
      }
    }
  }

  TORCH_INTERNAL_ASSERT(
      !missingRanks.empty() || !traceMap.empty(),
      "Trace shouldn't be empty while enabled GLOO_ASYNC_TIMEOUT_DEBUG");
  TORCH_INTERNAL_ASSERT(
      !thisCol.empty(),
      "Timeout rank [",
      myRank,
      "] must have collective tracking iteam in c10::Store trace");
  TORCH_INTERNAL_ASSERT(
      traceMap[thisSeq][myRank].second == kEventStart,
      "Timeout rank [",
      myRank,
      "] last trace item must be kEventStart. thisSeq = ",
      thisSeq,
      ", col = ",
      thisCol);

  report += c10::str(
      "\n\t - [", myRank, "] Timeout at collective: ", thisCol, ", #", thisSeq);

  if (!missingRanks.empty()) {
    report += analyzeMissingRanks(missingRanks);
  } else {
    report += analyzeLaggingRanks(traceMap);
    report += dumpSnapshot(traceMap);
  }

  return report;
}

inline std::string pickle_str(const c10::IValue& v) {
  std::vector<char> result;
  {
    auto writer = [&](const char* data, size_t size) {
      result.insert(result.end(), data, data + size);
    };
    torch::jit::Pickler pickler(
        writer, nullptr, nullptr, nullptr, nullptr, false);
    pickler.protocol();
    pickler.pushIValue(v);
    pickler.stop();
  }
  return std::string(result.begin(), result.end());
}

inline std::string get_python_cpp_trace() {
  // usage:
  // LOG(INFO) << "stacktrace: "
  //           << get_python_cpp_trace();
  // warn: might be slow in getting cpp traces
  // because of slow/broken addr2line
  // in different system libs
  std::shared_ptr<torch::CapturedTraceback> tb =
      torch::CapturedTraceback::gather(
          /*python=*/true, /*script=*/true, /*cpp=*/true);
  torch::SymbolizedTracebacks s_tbs = torch::symbolize({tb.get()});
  const auto& s_tb = s_tbs.tracebacks.at(0);
<<<<<<< HEAD
  std::stringstream oss;
  for (auto idx : c10::irange(s_tb.size())) {
    auto frame_id = s_tb[idx];
    const auto& frame = s_tbs.all_frames.at(frame_id);
    oss << "#" << idx << " " << frame.funcname << " from " << frame.filename
        << ":" << frame.lineno << '\n';
  }
  return oss.str();
=======
  constexpr auto TB_FMT_CSTR = FMT_COMPILE("#{} {} from {}:{}\n");
  fmt::memory_buffer buf;
  auto buf_iter = std::back_inserter(buf);
  for (auto idx : c10::irange(s_tb.size())) {
    auto frame_id = s_tb[idx];
    const auto& frame = s_tbs.all_frames.at(frame_id);
    fmt::format_to(
        buf_iter,
        TB_FMT_CSTR,
        idx,
        frame.funcname,
        frame.filename,
        frame.lineno);
  }
  return fmt::to_string(buf);
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

inline c10::Dict<c10::IValue, c10::IValue> new_dict() {
  return c10::Dict<c10::IValue, c10::IValue>(
      c10::AnyType::get(), c10::AnyType::get());
}

inline c10::List<c10::IValue> new_list() {
  return c10::List<c10::IValue>(c10::AnyType::get());
}

inline std::string ranks_str(const std::vector<uint64_t>& ranks) {
<<<<<<< HEAD
  std::string str;
  for (const auto& rank : ranks) {
    if (str.empty()) {
      str = std::to_string(rank);
    } else {
      str += ", " + std::to_string(rank);
    }
  }
  return c10::str("[", str, "]");
=======
  return fmt::format("[{}]", fmt::join(ranks, ", "));
>>>>>>> 5729657180 ([ROCm] Specialized binary elementwise broadcast kernel for mixed dtypes with float/bfloat16/half (#2791))
}

} // namespace c10d
