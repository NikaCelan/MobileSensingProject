package flwr.android_client

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import dev.flower.flower_tflite.FlowerClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.concurrent.ExecutionException

suspend fun readAssetLines(
    context: Context,
    fileName: String,
    call: suspend (Int, String) -> Unit
) {
    withContext(Dispatchers.IO) {
        BufferedReader(InputStreamReader(context.assets.open(fileName))).useLines {
            it.forEachIndexed { i, l -> launch { call(i, l) } }
        }
    }
}

/**
 * Load training data from disk.
 */
@Throws
suspend fun loadData(
    context: Context,
    flowerClient: FlowerClient<Float3DArray, FloatArray>,
    device_id: Int
) {
    var words = arrayOf("down", "left", "right", "up")
    for (w in words) {
        Log.d(TAG, "loading word $w");
        Log.d(TAG, "loading train");
        readAssetLines(context, "data/files${device_id}_${w}_train.txt") { index, line ->
            Log.d(TAG, "data/user0$device_id/$w/$line");
            addSample(context, flowerClient, "data/user0$device_id/$w/$line", true)
        }
        Log.d(TAG, "loading test");
        readAssetLines(context, "data/files${device_id}_${w}_test.txt") { index, line ->
            Log.d(TAG, "data/user0$device_id/$w/$line");
            addSample(context, flowerClient, "data/user0$device_id/$w/$line", false)
        }
    }
}

@Throws
private fun addSample(
    context: Context,
    flowerClient: FlowerClient<Float3DArray, FloatArray>,
    photoPath: String,
    isTraining: Boolean
) {
    val options = BitmapFactory.Options()
    options.inPreferredConfig = Bitmap.Config.ARGB_8888
    val bitmap = BitmapFactory.decodeStream(context.assets.open(photoPath), null, options)!!
    val sampleClass = getClass(photoPath)

    // get rgb equivalent and class
    val rgbImage = prepareImage(bitmap)

    // add to the list.
    try {
        flowerClient.addSample(rgbImage, classToLabel(sampleClass), isTraining)
    } catch (e: ExecutionException) {
        throw RuntimeException("Failed to add sample to model", e.cause)
    } catch (e: InterruptedException) {
        // no-op
    }
}

fun getClass(path: String): String {
    return path.split("/".toRegex()).dropLastWhile { it.isEmpty() }.toTypedArray()[2]
}

/**
 * Normalizes a camera image to [0; 1], cropping it
 * to size expected by the model and adjusting for camera rotation.
 */
private fun prepareImage(bitmap: Bitmap): Float3DArray {
    val normalizedRgb = Array(SPEC_HEIGHT) { Array(SPEC_WIDTH) { FloatArray(3) } }
    for (y in 0 until SPEC_HEIGHT) {
        for (x in 0 until SPEC_WIDTH) {
            val rgb = bitmap.getPixel(x, y)
            val r = (rgb shr 16 and LOWER_BYTE_MASK) * (1 / 255.0f)
            val g = (rgb shr 8 and LOWER_BYTE_MASK) * (1 / 255.0f)
            val b = (rgb and LOWER_BYTE_MASK) * (1 / 255.0f)
            normalizedRgb[y][x][0] = r
            normalizedRgb[y][x][1] = g
            normalizedRgb[y][x][2] = b
        }
    }
    return normalizedRgb
}

private const val TAG = "Load Data"
const val LOWER_BYTE_MASK = 0xFF


const val SPEC_WIDTH = 496
const val SPEC_HEIGHT = 369

val CLASSES = listOf(
    "up",
    "down",
    "left",
    "right"
)

fun classToLabel(className: String): FloatArray {
    return CLASSES.map {
        if (className == it) 1f else 0f
    }.toFloatArray()
}
