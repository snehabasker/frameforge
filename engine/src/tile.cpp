#include "frameforge/tile.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace frameforge {

namespace {

struct TileOrigin {
  int y0, x0, h, w;  // in LR pixels
};

std::vector<int> TileStarts(int extent, int tile_size, int overlap) {
  std::vector<int> starts;
  if (tile_size >= extent) {
    starts.push_back(0);
    return starts;
  }
  int stride = std::max(1, tile_size - overlap);
  int pos = 0;
  while (pos < extent) {
    int start = pos;
    if (start + tile_size > extent) start = std::max(0, extent - tile_size);
    starts.push_back(start);
    if (start + tile_size >= extent) break;
    pos += stride;
  }
  // De-duplicate (can happen when the last clamp repeats a start).
  starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
  return starts;
}

std::vector<TileOrigin> BuildTiles(int h, int w, int tile_size, int overlap) {
  std::vector<TileOrigin> tiles;
  for (int y0 : TileStarts(h, tile_size, overlap)) {
    for (int x0 : TileStarts(w, tile_size, overlap)) {
      int th = std::min(tile_size, h - y0);
      int tw = std::min(tile_size, w - x0);
      tiles.push_back({y0, x0, th, tw});
    }
  }
  return tiles;
}

Tensor Crop(const Tensor& src, int y0, int x0, int h, int w) {
  Tensor out(1, src.c(), h, w);
  for (int c = 0; c < src.c(); ++c) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        out.at(0, c, y, x) = src.at(0, c, y0 + y, x0 + x);
      }
    }
  }
  return out;
}

// Linear feather: weight 1.0 in the tile's interior, ramping down to a
// small positive floor over `overlap_hr` pixels approaching an edge that
// is NOT the true frame boundary (frame-boundary edges have no
// neighboring tile to blend with, so they stay at full weight).
float EdgeWeight(int pos, int tile_extent, int overlap_hr, bool touches_frame_start,
                  bool touches_frame_end) {
  float w = 1.0f;
  if (!touches_frame_start && overlap_hr > 0) {
    w = std::min(w, std::min(1.0f, static_cast<float>(pos + 1) / overlap_hr));
  }
  if (!touches_frame_end && overlap_hr > 0) {
    int dist_from_end = tile_extent - pos;
    w = std::min(w, std::min(1.0f, static_cast<float>(dist_from_end) / overlap_hr));
  }
  return std::max(w, 0.05f);  // floor avoids a zero-weight seam if overlap is tiny
}

}  // namespace

Tensor RunTiled(OrtModelRunner& runner, const Tensor& lr_rgb, const Tensor* lr_depth,
                 const Tensor* lr_mv, const Tensor* prev_hr_output, int tile_size, int overlap) {
  const int scale = runner.spec().scale;
  const int h = lr_rgb.h(), w = lr_rgb.w();

  if (tile_size <= 0 || (tile_size >= h && tile_size >= w)) {
    return runner.Run(lr_rgb, lr_depth, lr_mv, prev_hr_output);
  }

  auto tiles = BuildTiles(h, w, tile_size, overlap);
  const int out_h = h * scale, out_w = w * scale;

  Tensor accum(1, 3, out_h, out_w);
  Tensor weight_sum(1, 1, out_h, out_w);

  for (const auto& t : tiles) {
    Tensor tile_rgb = Crop(lr_rgb, t.y0, t.x0, t.h, t.w);
    Tensor tile_depth, tile_mv, tile_prev;
    Tensor* depth_ptr = nullptr;
    Tensor* mv_ptr = nullptr;
    Tensor* prev_ptr = nullptr;
    if (lr_depth != nullptr) {
      tile_depth = Crop(*lr_depth, t.y0, t.x0, t.h, t.w);
      depth_ptr = &tile_depth;
    }
    if (lr_mv != nullptr) {
      tile_mv = Crop(*lr_mv, t.y0, t.x0, t.h, t.w);
      mv_ptr = &tile_mv;
    }
    if (prev_hr_output != nullptr) {
      tile_prev = Crop(*prev_hr_output, t.y0 * scale, t.x0 * scale, t.h * scale, t.w * scale);
      prev_ptr = &tile_prev;
    }

    Tensor tile_out = runner.Run(tile_rgb, depth_ptr, mv_ptr, prev_ptr);

    const int overlap_hr = overlap * scale;
    bool at_top = (t.y0 == 0);
    bool at_bottom = (t.y0 + t.h == h);
    bool at_left = (t.x0 == 0);
    bool at_right = (t.x0 + t.w == w);

    for (int y = 0; y < tile_out.h(); ++y) {
      float wy = EdgeWeight(y, tile_out.h(), overlap_hr, at_top, at_bottom);
      for (int x = 0; x < tile_out.w(); ++x) {
        float wx = EdgeWeight(x, tile_out.w(), overlap_hr, at_left, at_right);
        float weight = wy * wx;
        int oy = t.y0 * scale + y;
        int ox = t.x0 * scale + x;
        for (int c = 0; c < 3; ++c) {
          accum.at(0, c, oy, ox) += tile_out.at(0, c, y, x) * weight;
        }
        weight_sum.at(0, 0, oy, ox) += weight;
      }
    }
  }

  Tensor result(1, 3, out_h, out_w);
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      float wsum = std::max(weight_sum.at(0, 0, y, x), 1e-8f);
      for (int c = 0; c < 3; ++c) {
        result.at(0, c, y, x) = accum.at(0, c, y, x) / wsum;
      }
    }
  }
  return result;
}

}  // namespace frameforge
