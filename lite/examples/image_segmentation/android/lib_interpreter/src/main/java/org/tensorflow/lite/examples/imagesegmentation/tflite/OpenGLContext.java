package org.tensorflow.lite.examples.imagesegmentation.tflite;

import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.os.Build;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import androidx.annotation.Nullable;

import java.util.HashMap;
import java.util.Map;
import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.egl.EGLSurface;

public class OpenGLContext {
    private static final String TAG = "OpenGLContext";

    public static final int EGL_CONTEXT_CLIENT_VERSION = 0x3098;
    public static final int EGL_OPENGL_ES2_BIT = 0x4;
    public static final int EGL_OPENGL_ES3_BIT_KHR = 0x00000040;
    public static final int EGL_DRAW = 12377;
    public static final int EGL_READ = 12378;

    public static final int EGL14_API_LEVEL = android.os.Build.VERSION_CODES.JELLY_BEAN_MR1;

    private EGL10 egl;
    private EGLDisplay eglDisplay = EGL10.EGL_NO_DISPLAY;
    private EGLConfig eglConfig = null;
    private EGLContext eglContext = EGL10.EGL_NO_CONTEXT;
    private int[] singleIntArray;  // reuse this instead of recreating it each time
    private int glVersion;
    private long nativeEglContext = 0;
    private android.opengl.EGLContext egl14Context = null;
    private EGLSurface surface = null;

    public OpenGLContext() {
        singleIntArray = new int[1];
        egl = (EGL10) EGLContext.getEGL();
        eglDisplay = egl.eglGetDisplay(EGL10.EGL_DEFAULT_DISPLAY);
        if (eglDisplay == EGL10.EGL_NO_DISPLAY) {
            throw new RuntimeException("eglGetDisplay failed");
        }
        int[] version = new int[2];
        if (!egl.eglInitialize(eglDisplay, version)) {
            throw new RuntimeException("eglInitialize failed");
        }

        EGLContext realParentContext = EGL10.EGL_NO_CONTEXT;

        // Try to create an OpenGL ES 3 context first, then fall back on ES 2.
        try {
            createContext(realParentContext, 3);
            glVersion = 3;
        } catch (RuntimeException e) {
            Log.w(TAG, "could not create GLES 3 context: " + e);
        }

        surface = createOffscreenSurface(1, 1);
    }

    /** Returns the managed {@link EGLContext} */
    public EGLContext getContext() {
        return eglContext;
    }

    public void makeCurrent() {
        if (!egl.eglMakeCurrent(eglDisplay, surface, surface, eglContext)) {
            throw new RuntimeException("eglMakeCurrent failed");
        }
    }

    public void makeNothingCurrent() {
        if (!egl.eglMakeCurrent(
                eglDisplay, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_CONTEXT)) {
            throw new RuntimeException("eglMakeCurrent failed");
        }
    }

    public EGLSurface createOffscreenSurface(int width, int height) {
        int[] surfaceAttribs = {EGL10.EGL_WIDTH, width, EGL10.EGL_HEIGHT, height, EGL10.EGL_NONE};
        EGLSurface eglSurface = egl.eglCreatePbufferSurface(eglDisplay, eglConfig, surfaceAttribs);
        if (eglSurface == null) {
            throw new RuntimeException("surface was null");
        }
        return eglSurface;
    }

    /** Releases the resources held by this manager. */
    public void release() {
        if (eglDisplay != EGL10.EGL_NO_DISPLAY) {
            // Android is unusual in that it uses a reference-counted EGLDisplay.  So for
            // every eglInitialize() we need an eglTerminate().
            egl.eglMakeCurrent(
                    eglDisplay, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_SURFACE, EGL10.EGL_NO_CONTEXT);
            egl.eglDestroyContext(eglDisplay, eglContext);
            egl.eglTerminate(eglDisplay);
        }

        eglDisplay = EGL10.EGL_NO_DISPLAY;
        eglContext = EGL10.EGL_NO_CONTEXT;
        eglConfig = null;
    }

    /** Releases an {@link EGLSurface}. */
    public void releaseSurface(EGLSurface eglSurface) {
        egl.eglDestroySurface(eglDisplay, eglSurface);
    }

    private void createContext(
            EGLContext parentContext, int glVersion) {
        eglConfig = getConfig(glVersion, null);
        if (eglConfig == null) {
            throw new RuntimeException("Unable to find a suitable EGLConfig");
        }
        // Try to create an OpenGL ES 3 context first.
        int[] contextAttrs = {EGL_CONTEXT_CLIENT_VERSION, glVersion, EGL10.EGL_NONE};
        eglContext = egl.eglCreateContext(eglDisplay, eglConfig, parentContext, contextAttrs);
        if (eglContext == null || eglContext == EGL10.EGL_NO_CONTEXT) {
            int error = egl.eglGetError();
            throw new RuntimeException(
                    "Could not create GL context: EGL error: 0x"
                            + Integer.toHexString(error)
                            + (error == EGL10.EGL_BAD_CONTEXT
                            ? ": parent context uses a different version of OpenGL"
                            : ""));
        }
    }

    private EGLConfig getConfig(int glVersion, @Nullable int[] additionalConfigAttributes) {
        int[] baseAttribList = {
                EGL10.EGL_RED_SIZE, 8,
                EGL10.EGL_GREEN_SIZE, 8,
                EGL10.EGL_BLUE_SIZE, 8,
                EGL10.EGL_ALPHA_SIZE, 8,
                EGL10.EGL_DEPTH_SIZE, 16,
                EGL10.EGL_RENDERABLE_TYPE, glVersion == 3 ? EGL_OPENGL_ES3_BIT_KHR : EGL_OPENGL_ES2_BIT,
                EGL10.EGL_SURFACE_TYPE, EGL10.EGL_PBUFFER_BIT | EGL10.EGL_WINDOW_BIT,
                EGL10.EGL_NONE
        };
        int[] attribList = baseAttribList;
        // First count the matching configs. Note that eglChooseConfig will return configs that
        // match *or exceed* the requirements, and will put the ones that exceed first!
        int[] numConfigs = singleIntArray;
        if (!egl.eglChooseConfig(eglDisplay, attribList, null, 0, numConfigs)) {
            throw new IllegalArgumentException("eglChooseConfig failed");
        }

        if (numConfigs[0] <= 0) {
            throw new IllegalArgumentException("No configs match requested attributes");
        }

        EGLConfig[] configs = new EGLConfig[numConfigs[0]];
        if (!egl.eglChooseConfig(eglDisplay, attribList, configs, configs.length, numConfigs)) {
            throw new IllegalArgumentException("eglChooseConfig#2 failed");
        }

        // Try to find a config that matches our bit sizes exactly.
        EGLConfig bestConfig = null;
        for (EGLConfig config : configs) {
            int r = findConfigAttrib(config, EGL10.EGL_RED_SIZE, 0);
            int g = findConfigAttrib(config, EGL10.EGL_GREEN_SIZE, 0);
            int b = findConfigAttrib(config, EGL10.EGL_BLUE_SIZE, 0);
            int a = findConfigAttrib(config, EGL10.EGL_ALPHA_SIZE, 0);
            if ((r == 8) && (g == 8) && (b == 8) && (a == 8)) {
                bestConfig = config;
                break;
            }
        }
        if (bestConfig == null) {
            bestConfig = configs[0];
        }

        return bestConfig;
    }

    private int findConfigAttrib(EGLConfig config, int attribute, int defaultValue) {
        if (egl.eglGetConfigAttrib(eglDisplay, config, attribute, singleIntArray)) {
            return singleIntArray[0];
        }
        return defaultValue;
    }

}
