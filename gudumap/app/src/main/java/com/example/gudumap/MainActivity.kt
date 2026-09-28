package com.example.gudumap

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.lifecycle.viewmodel.compose.viewModel

import com.example.gudumap.ui.screens.NavigationScreen
import com.example.gudumap.ui.screens.SplashScreen
import com.example.gudumap.ui.theme.GudumapTheme
import com.example.gudumap.viewmodel.NavigationViewModel

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        // System-level cold-start splash (Theme.App.Starting, navy background only) --
        // must be installed before super.onCreate() per the AndroidX SplashScreen API.
        // It hands off to Compose's own SplashScreen.kt as soon as the first frame below
        // is drawn, so there is no gap/white-flash between the two.
        installSplashScreen()
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()

        setContent {

            GudumapTheme {

                // Created here, at the top, unconditionally -- NOT inside the splash/main
                // switch below -- so NavigationViewModel's own init{} block
                // (navigationEngine.start(), which registers sensors/location) begins
                // immediately on the first frame, whether or not the splash is currently
                // showing on top of it.
                val navViewModel: NavigationViewModel = viewModel(factory = NavigationViewModel.Factory)
                var showSplash by remember { mutableStateOf(true) }

                Box(modifier = Modifier.fillMaxSize()) {
                    // Composed unconditionally, from the very first frame: this starts
                    // NavigationScreen's own permission-check/request flow and begins
                    // receiving real navViewModel state updates immediately, in parallel
                    // with the splash overlay below -- the splash never delays real
                    // initialization, it only sits visually on top of it for a moment.
                    NavigationScreen(navViewModel = navViewModel)

                    AnimatedVisibility(
                        visible = showSplash,
                        exit = fadeOut(animationSpec = tween(durationMillis = 350))
                    ) {
                        SplashScreen(onFinished = { showSplash = false })
                    }
                }

            }
        }
    }
}
