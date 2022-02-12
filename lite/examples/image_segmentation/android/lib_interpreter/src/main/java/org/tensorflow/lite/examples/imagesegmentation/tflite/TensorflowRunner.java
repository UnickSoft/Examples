package org.tensorflow.lite.examples.imagesegmentation.tflite;

public class TensorflowRunner
{
    static
    {
        System.loadLibrary("tensorflowlite");
        System.loadLibrary("native-lib");
    }

    public void init(byte[] data) {
        nativeInstance = nativeInit(data);
    }
    public void destroy() {
        nativeDestroy(nativeInstance);
        nativeInstance = 0;
    }

    public int inputsNum() {
        return nativeInputsNum(nativeInstance);
    }
    public int[]  getInputsDim(int index) {
        return nativeGetInputsDim(nativeInstance, index);
    }

    public int    outputsNum() {
        return nativeOutputsNum(nativeInstance);
    }
    public int[]  getOutputsDim(int index) {
        return nativeGetOutputsDim(nativeInstance, index);
    }

    public void   bindInput(int index, int ssboId) {
        nativeBindInput(nativeInstance, index, ssboId);
    }
    public void   bindOutput(int index, int ssboId) {
        nativeBindOutput(nativeInstance, index, ssboId);
    }

    public boolean run() {
        return nativeRun(nativeInstance);
    }

    private native long nativeInit(byte[] data);
    private native void nativeDestroy(long instance);

    private native int    nativeInputsNum(long instance);
    private native int[]  nativeGetInputsDim(long instance, int index);

    private native int    nativeOutputsNum(long instance);
    private native int[]  nativeGetOutputsDim(long instance, int index);

    private native void   nativeBindInput(long instance, int index, int ssboId);
    private native void   nativeBindOutput(long instance, int index, int ssboId);

    private native boolean nativeRun(long instance);

    private long nativeInstance = 0;
}
