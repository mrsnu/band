package org.mrsnu.band.example;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;

import org.mrsnu.band.BackendType;
import org.mrsnu.band.Band;
import org.mrsnu.band.ConfigBuilder;
import org.mrsnu.band.Config;
import org.mrsnu.band.CpuMaskFlags;
import org.mrsnu.band.Device;
import org.mrsnu.band.Engine;
import org.mrsnu.band.Model;
import org.mrsnu.band.SchedulerType;
import org.mrsnu.band.SubgraphPreparationType;
import org.mrsnu.band.Tensor;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Band.init();
        ConfigBuilder b = new ConfigBuilder();
        Engine engine = null;

        b.addPlannerLogPath("/data/data/org.mrsnu.band.example/log.tsv");
        b.addSchedulers(new SchedulerType[] {SchedulerType.ROUND_ROBIN});
        b.addMinimumSubgraphSize(7);
        b.addSubgraphPreparationType(SubgraphPreparationType.MERGE_UNIT_SUBGRAPH);
        b.addCPUMask(CpuMaskFlags.ALL);
        b.addPlannerCPUMask(CpuMaskFlags.PRIMARY);
        b.addWorkers(new Device[] {Device.CPU, Device.GPU, Device.DSP, Device.NPU});
        b.addWorkerNumThreads(new int[] {1, 1, 1, 1});
        b.addWorkerCPUMasks(new CpuMaskFlags[] {CpuMaskFlags.BIG, CpuMaskFlags.LITTLE, CpuMaskFlags.ALL, CpuMaskFlags.ALL});
        b.addSmoothingFactor(0.1f);
        b.addProfileDataPath("/data/data/org.mrsnu.band.example/profile.json");
        b.addOnline(true);
        b.addNumWarmups(1);
        b.addNumRuns(1);
        b.addAllowWorkSteal(true);
        b.addAvailabilityCheckIntervalMs(30000);
        b.addScheduleWindowSize(10);
        Config config = b.build();
        engine = new Engine(config);
        AssetManager am = getAssets();

        List<Tensor> inputTensors = new ArrayList<>();
        List<Tensor> outputTensors = new ArrayList<>();
        try {
            Model model = new Model(BackendType.TFLITE, loadModelFile(getAssets(), "add.tflite"));
            engine.registerModel(model);

            int numInputs = engine.getNumInputTensors(model);
            int numOutputs = engine.getNumOutputTensors(model);
            for (int i = 0; i < numInputs; i++) {
                inputTensors.add(engine.createInputTensor(model, i));
            }
            for (int i = 0; i < numOutputs; i++) {
                outputTensors.add(engine.createOutputTensor(model, i));
            }
            Log.i("BAND_JAVA", String.format("Successfully created input (%d) / output (%d) tensors.", numInputs, numOutputs));

            int[] data = { 100, 200 };
            ByteBuffer inputByteBuffer = ByteBuffer.allocate(data.length * 4);
            IntBuffer inputIntBuffer = inputByteBuffer.asIntBuffer();
            inputIntBuffer.put(data);
            byte[] array = inputByteBuffer.array();
            inputTensors.get(0).setData(array);
            engine.requestSync(model, inputTensors, outputTensors);
            byte[] outputArray = outputTensors.get(0).getData();
            Log.i("RESULT", String.format("%d", outputArray.length));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}