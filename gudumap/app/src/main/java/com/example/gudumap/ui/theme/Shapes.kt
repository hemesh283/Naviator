package com.example.gudumap.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Shapes
import androidx.compose.ui.unit.dp

// Larger, softer "squircle"-style rounding for the iOS-glass aesthetic -- roughly
// 20-28dp across the size scale Material3 components actually pull from
// (extraSmall..extraLarge), versus Material3's stock ~4-16dp defaults.
val GudumapShapes = Shapes(
    extraSmall = RoundedCornerShape(20.dp),
    small = RoundedCornerShape(22.dp),
    medium = RoundedCornerShape(24.dp),
    large = RoundedCornerShape(26.dp),
    extraLarge = RoundedCornerShape(28.dp),
)
