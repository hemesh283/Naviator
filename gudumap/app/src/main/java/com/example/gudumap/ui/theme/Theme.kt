package com.example.gudumap.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

// Deep navy / electric cyan / violet glassmorphic ColorScheme, enforced as the app's only
// theme rather than following system light/dark.
//
// Confirmed before hard-coding this: grepped the whole app for `isSystemInDarkTheme` and
// for any call site passing `GudumapTheme(darkTheme = ...)` -- the only hit for either was
// this file's own old default parameter, and `MainActivity.kt`'s sole call site
// (`GudumapTheme { ... }`) never overrode it. Color.kt's old `LightColorScheme` was also
// still the untouched stock Material3 template, never designed against -- so there was no
// real light-mode experience anyone depended on to preserve.
//
// Dynamic (Material You / wallpaper-derived) color support is also deliberately removed
// here, not just left at its old default: on API 31+ it would replace every color decided
// in Color.kt with whatever the device wallpaper happens to generate, which defeats the
// point of a deliberate navy/cyan/violet brand identity for a screening demo.
private val GudumapDarkColorScheme = darkColorScheme(
    primary = CyanPrimary,
    onPrimary = OnCyanPrimary,
    primaryContainer = CyanPrimaryContainer,
    onPrimaryContainer = OnCyanPrimaryContainer,

    secondary = VioletSecondary,
    onSecondary = OnVioletSecondary,
    secondaryContainer = VioletSecondaryContainer,
    onSecondaryContainer = OnVioletSecondaryContainer,

    tertiary = TertiaryPink,
    onTertiary = OnTertiaryPink,
    tertiaryContainer = TertiaryPinkContainer,
    onTertiaryContainer = OnTertiaryPinkContainer,

    background = NavyBase,
    onBackground = TextPrimary,

    surface = NavySurface,
    onSurface = TextPrimary,
    surfaceVariant = NavySurfaceVariant,
    onSurfaceVariant = TextMuted,

    outline = Outline,

    error = ErrorRed,
    onError = OnErrorRed,
    errorContainer = ErrorRedContainer,
    onErrorContainer = OnErrorRedContainer,
)

@Composable
fun GudumapTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = GudumapDarkColorScheme,
        typography = Typography,
        shapes = GudumapShapes,
        content = content
    )
}
