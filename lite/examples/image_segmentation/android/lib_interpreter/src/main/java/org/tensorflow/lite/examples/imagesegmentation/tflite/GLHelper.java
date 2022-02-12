package org.tensorflow.lite.examples.imagesegmentation.tflite;

import android.content.Context;
import android.graphics.Bitmap;
import android.opengl.GLES20;
import android.opengl.GLES31;
import android.opengl.GLUtils;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;

public class GLHelper {

    private static final String TAG = "GLHelper";

    public static String TextureToBufferShaderText = "#version 310 es\n" +
            "layout(local_size_x = 8, local_size_y = 8) in;\n" +
            "layout(binding = 0) uniform sampler2D u_Texture0;\n" +
            "layout(std430) buffer;\n" +
            "layout(binding = 1) buffer Output { float elements[]; } output_data;\n" +
            "uniform int u_width;\n" +
            "uniform int u_height;\n" +
            "void main() \n" +
            "{\n" +
            "ivec2 gid = ivec2(gl_GlobalInvocationID.xy);\n" +
            "if (gid.x >= u_width || gid.y >= u_height) return;\n" +
            "vec3 pixel = texelFetch(u_Texture0, gid, 0).xyz;\n" +
            "int linear_index = 3 * (gid.y * u_width + gid.x);\n" +
            "output_data.elements[linear_index + 0] = pixel.x;\n" +
            "output_data.elements[linear_index + 1] = pixel.y;\n" +
            "output_data.elements[linear_index + 2] = pixel.z;\n" +
            "//output_data.elements[linear_index + 3] = 0.0;\n" +
            "}\n";

    public static int createTexture() {
        final int[] textureHandle = new int[1];

        GLES20.glGenTextures(1, textureHandle, 0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureHandle[0]);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        checkGlError("createTexture");

        return textureHandle[0];
    }

    public static void destroyTexture(int textureId) {
        GLES20.glDeleteTextures(1, new int[]{textureId}, 0);
    }

    public static void bitmapToTexture(Bitmap image, int texture) {
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texture);
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, image, 0);
    }

    public static int createComputerProgram(String computerSource) {
        int computeShader = loadShader(GLES31.GL_COMPUTE_SHADER, computerSource);
        if (computeShader == 0) {
            return 0;
        }

        int program = GLES20.glCreateProgram();
        if (program == 0) {
            Log.v(TAG, "Could not create program");
        }

        if (computeShader > 0) {
            GLES20.glAttachShader(program, computeShader);
        }

        GLES20.glLinkProgram(program);
        int[] linkStatus = new int[1];
        GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0);
        if (linkStatus[0] != GLES20.GL_TRUE) {
            Log.v(TAG, "Could not link program: " + GLES20.glGetProgramInfoLog(program));
            GLES20.glDeleteProgram(program);
            program = 0;
        }
        checkGlError("createComputerProgram");

        return program;
    }

    public static int loadShader(int shaderType, String source) {
        int shader = GLES20.glCreateShader(shaderType);
        GLES20.glShaderSource(shader, source);
        GLES20.glCompileShader(shader);
        int[] compiled = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            Log.v(TAG, "Could not compile shader " + shaderType + ":" +
                    GLES20.glGetShaderInfoLog(shader));
            GLES20.glDeleteShader(shader);
            shader = 0;
        }
        return shader;
    }

    public static int getUniformLocation (int shader, String uniform) {
        return GLES20.glGetUniformLocation(shader, uniform);
    }

    public static void checkGlError(String msg) {
        int error = GLES20.glGetError();
        if (error != GLES20.GL_NO_ERROR) {
            throw new RuntimeException(TAG + " " + msg + ": GL error: 0x" + Integer.toHexString(error));
        }
    }

    public static int initializeShaderBuffer(int size) {
        int[] id = new int[1];
        GLES31.glGenBuffers(id.length, id, 0);
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, id[0]);

        GLES31.glBufferData(GLES31.GL_SHADER_STORAGE_BUFFER, size, null, GLES31.GL_STREAM_COPY);
        GLES31.glBindBufferBase(GLES31.GL_SHADER_STORAGE_BUFFER, 1, id[0]);
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0);// unbind
        return id[0];
    }

    public static void destroyBuffer (int buffer) {
        GLES20.glDeleteBuffers(1, new int[]{buffer}, 0);
    }

    public static void copyTextureToBuffer(int textureId, int texWidth, int texHeight, int bufferId, int bufferSize) {
        GLES31.glActiveTexture(GLES31.GL_TEXTURE0 + 0);
        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, textureId);

        GLES31.glBindBufferRange(GLES31.GL_SHADER_STORAGE_BUFFER, 1, bufferId, 0, bufferSize);
        GLES31.glDispatchCompute(texWidth / 8, texHeight / 8, 1);  // these are work group sizes
        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, 0);  // unbind
        GLES31.glBindTexture(GLES31.GL_TEXTURE_2D, 0);  // unbind
        GLES31.glMemoryBarrier(GLES31.GL_ALL_BARRIER_BITS);
        checkGlError("copyTextureToBuffer");
    }

    public static void getReadGPUBufferData(ByteBuffer cpuBuffer, int bufferId, int size)
    {
        GLES20.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, bufferId);
        checkGlError("glBindBuffer");

        final ByteBuffer buffer = (ByteBuffer) GLES31.glMapBufferRange(
                GLES31.GL_SHADER_STORAGE_BUFFER,
                0,
                size,
                GLES31.GL_MAP_READ_BIT);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        checkGlError("glMapBufferRange");
        cpuBuffer.put(buffer);

        // unmap the buffers
        GLES31.glUnmapBuffer(GLES31.GL_SHADER_STORAGE_BUFFER);
        checkGlError("glUnmapBuffer");

        GLES31.glBindBuffer(GLES31.GL_SHADER_STORAGE_BUFFER, GLES20.GL_NONE);
        checkGlError("glBindBuffer");
    }
}
