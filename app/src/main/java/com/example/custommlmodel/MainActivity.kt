package com.example.custommlmodel

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.result.launch
import androidx.annotation.RequiresApi
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material.Button
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Surface
import androidx.compose.material.Text
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.custommlmodel.ml.ModelUnquant
import com.example.custommlmodel.ui.theme.CustomMLModelTheme
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer
import java.nio.ByteOrder

const val image_size = 224
class MainActivity : ComponentActivity() {
    @RequiresApi(Build.VERSION_CODES.P)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val imageBitmap = remember {
                mutableStateOf<Bitmap?>(
                    null
                )
            }
            val context = LocalContext.current
            val typePredicted = remember {
                mutableStateOf("")
            }
            val confidence = remember {
                mutableStateOf("")
            }
            val launcher = rememberLauncherForActivityResult(contract = ActivityResultContracts.TakePicturePreview()){
                imageBitmap.value = it
            }
            val launcher1 = rememberLauncherForActivityResult(contract = ActivityResultContracts.GetContent()){
                imageBitmap.value =  if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, it))
                } else {
                    MediaStore.Images.Media.getBitmap(context.contentResolver, it)
                }
            }

            
            CustomMLModelTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colors.background
                ) {
                    Column(
                        modifier = Modifier.fillMaxSize(),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        imageBitmap.value?.let {
                            Image(
                                bitmap = it.asImageBitmap(),
                                contentDescription = "Image" ,
                                contentScale = ContentScale.FillBounds,
                                modifier = Modifier.size(300.dp)
                            )
                        }
                        Text(text = "Type : ${typePredicted.value}")
                        Text(text = "Confidence : ${confidence.value}")

                        Row(modifier = Modifier.fillMaxWidth()) {
                            Button(
                                onClick = {
                                    launcher.launch()
                                },
                                modifier = Modifier
                                    .weight(1f)
                                    .padding(all = 5.dp)
                            ) {
                                Text(text = "Take Picture")
                            }

                            Button(
                                onClick = {
                                    launcher1.launch("image/*")
                                },
                                modifier = Modifier
                                    .weight(1f)
                                    .padding(all = 5.dp)
                            ) {
                                Text(text = "Select Image")
                            }
                        }
                        
                        Button(onClick = {
                            if(imageBitmap.value != null){
                                classifyImage(
                                    imageBitmap.value!!,
                                    context,
                                    typePredicted,
                                    confidence
                                )
                            }else{
                                Toast.makeText(context,"Select Image or Take Picture",Toast.LENGTH_LONG).show()
                            }
                        }) {
                            Text(text = "Classify Image")
                        }
                    }
                }
            }
        }
    }

    fun classifyImage(
        image: Bitmap,
        context: Context,
        typePredicted: MutableState<String>,
        confidence: MutableState<String>
    ) {
        val imageBitmap = Bitmap.createScaledBitmap(image, image_size , image_size,false)
        val model = ModelUnquant.newInstance(context)

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
        val byteBuffer = ByteBuffer.allocateDirect(4* image_size* image_size*3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(image_size* image_size)
        imageBitmap.getPixels(intValues,0,imageBitmap.width,0,0,imageBitmap.width, imageBitmap.height)
        var pixel = 0
        for (i in 0 until image_size){
            for (j in 0 until image_size){
                val value = intValues[pixel++]
                byteBuffer.putFloat(((value shl 16)and 0xFF)*(1F/255F))
                byteBuffer.putFloat(((value shl 8)and 0xFF)*(1F/255F))
                byteBuffer.putFloat((value and 0xFF)*(1F/255F))
            }
        }

        inputFeature0.loadBuffer(byteBuffer)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        val confidenceArray = outputFeature0.floatArray.toList()
        val max : Float? = confidenceArray.maxOrNull()
        val index : Int = confidenceArray.indexOf(max)
        val classesNames = arrayListOf("Bird","Animal","Human")

        typePredicted.value = classesNames[index]
        confidence.value = max.toString()

        model.close()
    }
}