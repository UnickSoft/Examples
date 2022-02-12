/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.imagesegmentation.tflite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.opengl.GLES20
import android.opengl.GLES31
import android.os.SystemClock
import android.util.Log
import androidx.core.graphics.ColorUtils
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.imagesegmentation.utils.ImageUtils
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.random.Random

/**
 * Class responsible to run the Image Segmentation model. more information about the DeepLab model
 * being used can be found here:
 * https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html
 * https://www.tensorflow.org/lite/models/segmentation/overview
 * https://github.com/tensorflow/models/tree/master/research/deeplab
 *
 * Label names: 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
 * 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
 * 'sofa', 'train', 'tv'
 */
class ImageSegmentationModelExecutorGPUPass(context: Context) : ImageSegmentationModelExecutor(context) {

  protected var prepareGPUDataExecutionTime = 0L
  protected var gettingResultFromGPUExecutionTime = 0L

  // GPU native runner.
  private val nativeRunner = TensorflowRunner()
  private var inputTexture  = 0
  private var convertShader = 0
  private var shaderInputTexture = 0
  private var shaderWidth        = 0
  private var shaderHeight       = 0

  private var inputBuffer  = 0
  private var outputBuffer  = 0

  private var bufferBinded = false;

  private val openGlContext = OpenGLContext()

  init {
    // Init GPU model on thread with OpenGL context.
    openGlContext.makeCurrent()
    nativeRunner.init(getByteArrayFromByteBuffer(loadModelFile(context, imageSegmentationModel)))
    openGlContext.makeNothingCurrent()
  }

  override fun execute(data: Bitmap): ModelExecutionResult {
    try {
      openGlContext.makeCurrent()

      fullTimeExecutionTime = SystemClock.uptimeMillis()

      preprocessTime = SystemClock.uptimeMillis()
      val scaledBitmap = ImageUtils.scaleBitmapAndKeepRatio(data, imageSize, imageSize)

      val contentArray =
        ImageUtils.bitmapToByteBuffer(scaledBitmap, imageSize, imageSize, IMAGE_MEAN, IMAGE_STD)
      preprocessTime = SystemClock.uptimeMillis() - preprocessTime

      prepareGPUDataExecutionTime = SystemClock.uptimeMillis()
      prepareGpuData(scaledBitmap)
      assert(bufferBinded)
      prepareGPUDataExecutionTime = SystemClock.uptimeMillis() - prepareGPUDataExecutionTime

      imageSegmentationTime = SystemClock.uptimeMillis()
      nativeRunner.run()
      GLES31.glMemoryBarrier(GLES31.GL_ALL_BARRIER_BITS);
      imageSegmentationTime = SystemClock.uptimeMillis() - imageSegmentationTime
      Log.d(TAG, "Time to run the model $imageSegmentationTime")

      gettingResultFromGPUExecutionTime = SystemClock.uptimeMillis()
      GLHelper.getReadGPUBufferData(segmentationMasks, outputBuffer, 1 * imageSize * imageSize * NUM_CLASSES * 4)
      gettingResultFromGPUExecutionTime = SystemClock.uptimeMillis() - gettingResultFromGPUExecutionTime

      maskFlatteningTime = SystemClock.uptimeMillis()
      val (maskImageApplied, maskOnly, itemsFound) =
        convertBytebufferMaskToBitmap(
          segmentationMasks,
          imageSize,
          imageSize,
          scaledBitmap,
          segmentColors
        )
      maskFlatteningTime = SystemClock.uptimeMillis() - maskFlatteningTime
      Log.d(TAG, "Time to flatten the mask result $maskFlatteningTime")

      fullTimeExecutionTime = SystemClock.uptimeMillis() - fullTimeExecutionTime
      Log.d(TAG, "Total time execution $fullTimeExecutionTime")

      openGlContext.makeNothingCurrent();

      return ModelExecutionResult(
        maskImageApplied,
        scaledBitmap,
        maskOnly,
        formatExecutionLog(),
        itemsFound
      )
    } catch (e: Exception) {
      val exceptionLog = "something went wrong: ${e.message}"
      Log.d(TAG, exceptionLog)

      val emptyBitmap = ImageUtils.createEmptyBitmap(imageSize, imageSize)
      return ModelExecutionResult(
        emptyBitmap,
        emptyBitmap,
        emptyBitmap,
        exceptionLog,
        HashMap<String, Int>()
      )
    }
  }

  fun prepareGpuData(data: Bitmap)  {
    if (inputTexture == 0) {
      inputTexture = GLHelper.createTexture();
    }

    if (convertShader == 0) {
      convertShader = GLHelper.createComputerProgram(GLHelper.TextureToBufferShaderText)
      shaderInputTexture = GLHelper.getUniformLocation (convertShader, "u_Texture0");
      shaderWidth        = GLHelper.getUniformLocation (convertShader, "u_width");
      shaderHeight       = GLHelper.getUniformLocation (convertShader, "u_height");
      GLHelper.checkGlError("create convert shader")
    }

    val inputBufferSize = imageSize * imageSize * 3 * 4
    if (inputBuffer == 0) {
      inputBuffer = GLHelper.initializeShaderBuffer(inputBufferSize)
    }

    if (outputBuffer == 0) {
      outputBuffer = GLHelper.initializeShaderBuffer(1 * imageSize * imageSize * NUM_CLASSES * 4)
    }

    GLHelper.bitmapToTexture(data, inputTexture);

    GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
    GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, inputTexture)

    GLES20.glUseProgram(convertShader)
    GLES20.glUniform1i(shaderInputTexture, 0)
    GLES20.glUniform1i(shaderWidth,  imageSize)
    GLES20.glUniform1i(shaderHeight, imageSize)

    GLHelper.copyTextureToBuffer(inputTexture, imageSize, imageSize, inputBuffer, inputBufferSize)

    if (!bufferBinded) {
      nativeRunner!!.bindInput(0, inputBuffer)
      nativeRunner!!.bindOutput(0, outputBuffer)
      bufferBinded = true
    }
  }

  private fun formatExecutionLog(): String {
    val sb = StringBuilder()
    sb.append("Input Image Size: $imageSize x $imageSize\n")
    sb.append("GPU is enabled and pass GPU Buffers to TFL\n")
    sb.append("Number of threads: $numberThreads\n")
    sb.append("Pre-process execution time: $preprocessTime ms\n")
    sb.append("Manual push data to GPU time: $prepareGPUDataExecutionTime ms\n")
    sb.append("Model execution time: $imageSegmentationTime ms\n")
    sb.append("Manual pull data from GPU time: $gettingResultFromGPUExecutionTime ms\n")
    sb.append("Mask flatten time: $maskFlatteningTime ms\n")
    sb.append("Full execution time: $fullTimeExecutionTime ms\n")
    return sb.toString()
  }

  override fun close() {
    super.close()

    GLHelper.destroyTexture(inputTexture)
    GLHelper.destroyBuffer(inputBuffer)
    GLHelper.destroyBuffer(outputBuffer)
    nativeRunner.destroy()
  }

  private fun getByteArrayFromByteBuffer(byteBuffer: ByteBuffer): ByteArray {
    val bytesArray = ByteArray(byteBuffer.remaining())
    byteBuffer[bytesArray, 0, bytesArray.size]
    return bytesArray
  }

  companion object {
    private const val TAG = "SegmentationInterGPUPass"
  }
}
