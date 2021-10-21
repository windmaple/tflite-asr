package org.tensorflow.lite.examples.asr;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.MediaPlayer;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

import com.jlibrosa.audio.JLibrosa;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    private MappedByteBuffer tfLiteModel;
    private Interpreter tfLite;

    private Spinner audioClipSpinner;
    private Button transcribeButton;
    private Button playAudioButton;
    private TextView resultTextview;

    private String wavFilename;
    private MediaPlayer mediaPlayer = new MediaPlayer();

    private final static String TAG = "TfLiteASRDemo";
    private final static int SAMPLE_RATE = 16000;
    private final static int DEFAULT_AUDIO_DURATION = -1;
    private final static String[] WAV_FILENAMES = {"audio_clip_1.wav", "audio_clip_2.wav", "audio_clip_3.wav"};
    private final static String TFLITE_FILE = "CONFORMER.tflite";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        JLibrosa jLibrosa = new JLibrosa();

        audioClipSpinner = findViewById(R.id.audio_clip_spinner);
        ArrayAdapter<String>adapter = new ArrayAdapter<String>(MainActivity.this,
                android.R.layout.simple_spinner_item, WAV_FILENAMES);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        audioClipSpinner.setAdapter(adapter);
        audioClipSpinner.setOnItemSelectedListener(this);

        playAudioButton = findViewById(R.id.play);
        playAudioButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                try (AssetFileDescriptor assetFileDescriptor = getAssets().openFd(wavFilename)) {
                    mediaPlayer.reset();
                    mediaPlayer.setDataSource(assetFileDescriptor.getFileDescriptor(), assetFileDescriptor.getStartOffset(), assetFileDescriptor.getLength());
                    mediaPlayer.prepare();
                } catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
                mediaPlayer.start();
            }
        });

        transcribeButton = findViewById(R.id.recognize);
        resultTextview = findViewById(R.id.result);
        transcribeButton.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                try {
                    float audioFeatureValues[] = jLibrosa.loadAndRead(copyWavFileToCache(wavFilename), SAMPLE_RATE, DEFAULT_AUDIO_DURATION);

                    Object[] inputArray = {audioFeatureValues};
                    IntBuffer outputBuffer = IntBuffer.allocate(2000);

                    Map<Integer, Object> outputMap = new HashMap<>();
                    outputMap.put(0, outputBuffer);

                    tfLiteModel = loadModelFile(getAssets(), TFLITE_FILE);
                    Interpreter.Options tfLiteOptions = new Interpreter.Options();
                    tfLite = new Interpreter(tfLiteModel, tfLiteOptions);
                    tfLite.resizeInput(0, new int[] {audioFeatureValues.length});

                    tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

                    int outputSize = tfLite.getOutputTensor(0).shape()[0];
                    int[] outputArray = new int[outputSize];
                    outputBuffer.rewind();
                    outputBuffer.get(outputArray);
                    StringBuilder finalResult = new StringBuilder();
                    for (int i=0; i < outputSize; i++) {
                        char c = (char) outputArray[i];
                        if (outputArray[i] != 0) {
                            finalResult.append((char) outputArray[i]);
                        }
                    }
                    resultTextview.setText(finalResult.toString());
                } catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
            }
        });

    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View v, int position, long id) {
        wavFilename = WAV_FILENAMES[position];
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {
    }

    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private String copyWavFileToCache(String wavFilename) {
            File destinationFile = new File(getCacheDir() + wavFilename);
            if (!destinationFile.exists()) {
                try {
                    InputStream inputStream = getAssets().open(wavFilename);
                    int inputStreamSize = inputStream.available();
                    byte[] buffer = new byte[inputStreamSize];
                    inputStream.read(buffer);
                    inputStream.close();

                    FileOutputStream fileOutputStream = new FileOutputStream(destinationFile);
                    fileOutputStream.write(buffer);
                    fileOutputStream.close();
                } catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
            }

            return getCacheDir() + wavFilename;
    }
}