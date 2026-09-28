// Tiled inference (CLAUDE.md 7.3): splits a frame's LR spatial extent into
// overlapping tiles, runs the model per tile, and blends the overlaps back
// into one full HR output -- for LR inputs too large to run in one shot.
#pragma once

#include "frameforge/ort_session.hpp"
#include "frameforge/tensor.hpp"

namespace frameforge {

// Runs `runner` over `lr_rgb` (and, when the model needs them, `lr_depth`/
// `lr_mv`/`prev_hr_output`) in `tile_size` x `tile_size` LR tiles with
// `overlap` LR pixels of context on each side, linearly feathering the
// overlap regions back together at HR resolution. `prev_hr_output`, if
// given, is expected at full HR resolution already (the same tensor the
// untiled path would use) -- this function slices out each tile's HR
// region of it internally.
//
// With `tile_size <= 0`, runs untiled (single call to runner.Run()) --
// tests/tile_test.cpp checks tiled and untiled agree within tolerance for
// a real tile_size.
Tensor RunTiled(OrtModelRunner& runner, const Tensor& lr_rgb, const Tensor* lr_depth,
                 const Tensor* lr_mv, const Tensor* prev_hr_output, int tile_size, int overlap);

}  // namespace frameforge
