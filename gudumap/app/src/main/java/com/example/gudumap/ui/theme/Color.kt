package com.example.gudumap.ui.theme

import androidx.compose.ui.graphics.Color

// ============================================================
// Deep navy / electric cyan / violet glassmorphic palette.
//
// Every foreground-on-background pairing below was checked against the
// WCAG 2.x relative-luminance contrast formula before being picked, not
// just eyeballed -- the ratio each color was actually chosen to clear is
// noted inline. Normal-size text targets >=4.5:1 (WCAG AA); anything
// explicitly marked "non-text" targets the looser >=3:1 UI-component
// minimum instead, since it never carries text.
// ============================================================

// --- Backgrounds ---
val NavyBase = Color(0xFF0A0E27)            // app background, near-black navy
val NavySurface = Color(0xFF12172E)         // card/panel base surface
val NavySurfaceVariant = Color(0xFF1A2142)  // subtly elevated surface (dividers, variant fills)
val NavyElevated = Color(0xFF161B38)        // glass-panel container tint

// --- Primary: electric cyan ---
val CyanPrimary = Color(0xFF22D3EE)
val OnCyanPrimary = Color(0xFF062024)              // 9.37:1 on CyanPrimary
val CyanPrimaryContainer = Color(0xFF0C5C73)
val OnCyanPrimaryContainer = Color(0xFFA5F3FC)     // 6.02:1 on CyanPrimaryContainer

// --- Secondary: violet / purple ---
val VioletSecondary = Color(0xFFA78BFA)
val OnVioletSecondary = Color(0xFF1E1338)          // 6.40:1 on VioletSecondary
val VioletSecondaryContainer = Color(0xFF4C2E8F)
val OnVioletSecondaryContainer = Color(0xFFD8CCFC) // 6.67:1 on VioletSecondaryContainer

// --- Tertiary: soft pink-violet accent (Material3 wants a 3rd accent role distinct
// from primary/secondary; kept in the same purple family per the brief) ---
val TertiaryPink = Color(0xFFF0ABFC)
val OnTertiaryPink = Color(0xFF3B0764)              // 8.52:1 on TertiaryPink
val TertiaryPinkContainer = Color(0xFF701A75)
val OnTertiaryPinkContainer = Color(0xFFFAD1FF)    // 7.47:1 on TertiaryPinkContainer

// --- Text ---
val TextPrimary = Color(0xFFE8EAF6)   // 15.86:1 on NavyBase, 14.76:1 on NavySurface
val TextMuted = Color(0xFF9CA3C9)     // 7.15:1 on NavySurface

// --- Status / telemetry semantics -- custom roles, not part of Material3's ColorScheme,
// for the accept/warn/reject-style readouts this app's screens already show (GNSS/ML
// gate/EKF status etc.). Picked from the same "400-weight vivid on near-black" family as
// the rest of the palette specifically because that family reads reliably on dark glass. ---
val StatusGood = Color(0xFF34D399)     // 9.20:1 on NavySurface
val StatusWarning = Color(0xFFFBBF24)  // 10.59:1 on NavySurface
val StatusError = Color(0xFFF87171)    // 6.39:1 on NavySurface, 6.87:1 on NavyBase
val OnStatus = NavyBase                // dark text/icon on top of any bright status color above

// --- Material3 error role (reuses StatusError so "error" reads as one consistent color
// everywhere instead of introducing a second, different red) ---
val ErrorRed = StatusError
val OnErrorRed = Color(0xFF450A0A)          // 5.84:1 on ErrorRed
val ErrorRedContainer = Color(0xFF7F1D1D)
val OnErrorRedContainer = Color(0xFFFECACA) // 6.93:1 on ErrorRedContainer

// --- Outline / borders ---
val Outline = Color(0xFF5B6699)             // non-text: 3.2:1 on NavySurface, 3.44:1 on NavyBase
val GlassEdgeHighlight = Color(0x24FFFFFF)  // translucent white hairline for glass-panel edges --
                                             // decorative only, not a meaningful boundary, so not
                                             // held to the 3:1 rule above
