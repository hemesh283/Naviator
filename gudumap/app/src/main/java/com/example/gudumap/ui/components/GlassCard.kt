package com.example.gudumap.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.unit.dp
import com.example.gudumap.ui.theme.GlassEdgeHighlight
import com.example.gudumap.ui.theme.NavyElevated

/**
 * A reusable "glass panel" surface for the navy/cyan/violet design system: a
 * semi-transparent tinted background with a subtle top-lit sheen gradient, a thin
 * translucent edge highlight, and a soft ambient shadow -- approximating the iOS
 * glassmorphism look.
 *
 * IMPORTANT -- this is a semi-transparent-scrim APPROXIMATION, not a real backdrop blur.
 * Whatever is actually behind this card is only tinted/dimmed by its own background
 * color, never blurred: Compose has no first-party blur-behind-content primitive.
 * `dev.chrisbanes.haze` (the real option for genuine backdrop blur) was deliberately NOT
 * added in this pass -- see this session's PROJECT_STATUS.md entry for the concrete
 * reasons (its recent releases have been actively churning their minCompileSdk
 * requirement -- one release needed compileSdk 37.2 and had to be walked back to 37.0 --
 * and this environment has no way to build or run the app on a device to confirm it
 * actually renders correctly and performantly here, which the task itself required
 * before adopting it). Callers only see `modifier`/`shape`/`content`, so swapping this
 * composable's internals for real Haze blur later should not require touching call sites.
 */
@Composable
fun GlassCard(
    modifier: Modifier = Modifier,
    shape: Shape = RoundedCornerShape(24.dp),
    // §56: tint opacity at the top of the panel's sheen gradient (the bottom is always 0.14
    // lower). Default 0.72f reproduces the original look exactly, so every existing caller is
    // unchanged; a caller that needs a slightly more solid, more legible frosted panel (e.g. the
    // details drawer, which sits over busy map tiles) can pass a higher value.
    opacity: Float = 0.72f,
    // §59: optional "frost" -- a faint milky-white sheen layered over the navy tint (strongest
    // at the top, fading down), plus a brighter edge hairline. Real backdrop blur isn't
    // available without adding the Haze library (see the note above), so this is the
    // lighting cue that makes a tinted panel read as frosted glass rather than dark tinted
    // plastic. 0f (default) = no frost, so every existing caller is unchanged.
    frost: Float = 0f,
    content: @Composable () -> Unit
) {
    val topAlpha = opacity.coerceIn(0f, 1f)
    val bottomAlpha = (topAlpha - 0.14f).coerceIn(0f, 1f)
    Box(
        modifier = modifier
            .shadow(
                elevation = 16.dp,
                shape = shape,
                ambientColor = Color.Black.copy(alpha = 0.35f),
                spotColor = Color.Black.copy(alpha = 0.35f)
            )
            .clip(shape)
            .background(
                Brush.verticalGradient(
                    colors = listOf(
                        NavyElevated.copy(alpha = topAlpha),
                        NavyElevated.copy(alpha = bottomAlpha)
                    )
                )
            )
            .then(
                if (frost > 0f) {
                    Modifier.background(
                        Brush.verticalGradient(
                            colors = listOf(
                                Color.White.copy(alpha = frost.coerceIn(0f, 1f)),
                                Color.White.copy(alpha = (frost * 0.35f).coerceIn(0f, 1f))
                            )
                        )
                    )
                } else {
                    Modifier
                }
            )
            .border(
                width = 1.dp,
                color = if (frost > 0f) Color.White.copy(alpha = 0.22f) else GlassEdgeHighlight,
                shape = shape
            )
    ) {
        content()
    }
}
