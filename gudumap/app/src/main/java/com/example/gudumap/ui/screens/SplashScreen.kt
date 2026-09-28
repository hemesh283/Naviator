package com.example.gudumap.ui.screens

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.gudumap.R
import com.example.gudumap.ui.theme.CyanPrimary
import com.example.gudumap.ui.theme.NavyBase
import kotlinx.coroutines.delay

/**
 * Purely-visual entrance screen (phase 4 rebrand). Shows the isolated Naviator pin mark
 * (the same `ic_launcher_foreground.png` asset used by the launcher icon, for one consistent
 * brand image) plus a recreated "Naviator" wordmark in the app's own typography, on the app's
 * real navy background -- a soft fade-in + gentle scale-up (short `tween`s, `FastOutSlowInEasing`,
 * nothing spring/bouncy, consistent with phase 3's animation language), total on-screen time
 * ~1.3s, comfortably inside the 1.2-1.8s window this phase asked for.
 *
 * Deliberately dumb: this composable owns no app state and does no initialization work of its
 * own. `MainActivity` composes the real `NavigationScreen` (and the `NavigationViewModel` that
 * drives it) unconditionally from the very first frame, in parallel underneath this -- this is
 * only ever painted on TOP of already-running real content and self-dismisses via `onFinished`,
 * it never gates when sensor/location/permission setup begins.
 */
@Composable
fun SplashScreen(onFinished: () -> Unit, modifier: Modifier = Modifier) {
    var animateIn by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        animateIn = true
        delay(1300L)
        onFinished()
    }

    val alpha by animateFloatAsState(
        targetValue = if (animateIn) 1f else 0f,
        animationSpec = tween(durationMillis = 550, easing = FastOutSlowInEasing),
        label = "splashAlpha"
    )
    val markScale by animateFloatAsState(
        targetValue = if (animateIn) 1f else 0.88f,
        animationSpec = tween(durationMillis = 550, easing = FastOutSlowInEasing),
        label = "splashScale"
    )

    Box(
        modifier = modifier
            .fillMaxSize()
            .background(NavyBase),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier.graphicsLayer { this.alpha = alpha },
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Image(
                painter = painterResource(id = R.drawable.ic_launcher_foreground),
                contentDescription = null,
                modifier = Modifier
                    .size(120.dp)
                    .scale(markScale)
            )
            Spacer(modifier = Modifier.height(20.dp))
            Text(
                text = "Naviator",
                style = MaterialTheme.typography.headlineMedium.copy(
                    fontWeight = FontWeight.Bold
                ),
                color = CyanPrimary,
                modifier = Modifier.scale(markScale)
            )
        }
    }
}
