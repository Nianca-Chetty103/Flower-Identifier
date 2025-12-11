package com.example.floweridentifier;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private Interpreter tflite;
    private Bitmap bitmap;
    private List<String> labels;

    private Button captureButton, selectButton, predictButton;
    private ImageView captureImage;
    private TextView label;

    private static final int PERMISSION_REQUEST_CODE = 100;
    private static final int SELECT_IMAGE_REQUEST = 10;
    private static final int CAPTURE_IMAGE_REQUEST = 12;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // UI elements
        captureButton = findViewById(R.id.Capture);
        selectButton = findViewById(R.id.Select);
        predictButton = findViewById(R.id.predict);
        captureImage = findViewById(R.id.image);
        label = findViewById(R.id.labeltext);

        checkPermissions();

        // Load model + labels
        try {
            MappedByteBuffer modelFile = FileUtil.loadMappedFile(this, "mobilenet_v1_1.0_224.tflite");
            tflite = new Interpreter(modelFile);

            labels = FileUtil.loadLabels(this, "labels.txt");

        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Error loading model or labels.", Toast.LENGTH_LONG).show();
        }

        // Open gallery
        selectButton.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
            intent.setType("image/*");
            startActivityForResult(intent, SELECT_IMAGE_REQUEST);
        });

        // Open camera
        captureButton.setOnClickListener(v -> {
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(intent, CAPTURE_IMAGE_REQUEST);
        });

        // Run prediction (FIXED)
        predictButton.setOnClickListener(v -> {

            if (bitmap == null) {
                Toast.makeText(this, "Select an image first!", Toast.LENGTH_SHORT).show();
                return;
            }

            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

            // Prepare normalized buffer (MobileNet format)
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
            inputBuffer.order(ByteOrder.nativeOrder());

            int[] pixels = new int[224 * 224];
            resizedBitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224);

            int pixelIndex = 0;
            for (int i = 0; i < 224; i++) {
                for (int j = 0; j < 224; j++) {
                    int pixel = pixels[pixelIndex++];

                    float r = ((pixel >> 16) & 0xFF);
                    float g = ((pixel >> 8) & 0xFF);
                    float b = (pixel & 0xFF);

                    // MobileNet normalization: [-1, 1]
                    inputBuffer.putFloat((r - 127.5f) / 127.5f);
                    inputBuffer.putFloat((g - 127.5f) / 127.5f);
                    inputBuffer.putFloat((b - 127.5f) / 127.5f);
                }
            }

            float[][] output = new float[1][1001];
            tflite.run(inputBuffer, output);

            int maxIndex = -1;
            float maxProb = -1f;

            for (int i = 0; i < output[0].length; i++) {
                if (output[0][i] > maxProb) {
                    maxProb = output[0][i];
                    maxIndex = i;
                }
            }

            if (maxIndex >= 0) {
                String result = labels.get(maxIndex);
                label.setText(result);
                Toast.makeText(this, "Prediction: " + result, Toast.LENGTH_LONG).show();
            } else {
                label.setText("Unknown");
            }
        });
    }


    // Permission check
    private void checkPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            boolean cameraGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
            boolean storageGranted = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;

            if (!cameraGranted || !storageGranted) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE},
                        PERMISSION_REQUEST_CODE);
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE) {
            boolean allGranted = true;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    allGranted = false;
                    break;
                }
            }
            if (!allGranted) {
                Toast.makeText(this, "Permissions are required to use camera and gallery.", Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && data != null) {
            if (requestCode == SELECT_IMAGE_REQUEST && data.getData() != null) {
                Uri uri = data.getData();
                try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                    captureImage.setImageBitmap(bitmap);
                    label.setText("");
                } catch (IOException e) {
                    e.printStackTrace();
                }
            } else if (requestCode == CAPTURE_IMAGE_REQUEST && data.getExtras() != null) {
                bitmap = (Bitmap) data.getExtras().get("data");
                if (bitmap != null) {
                    captureImage.setImageBitmap(bitmap);
                    label.setText("");
                }
            }
        }
    }
}
