package org.mrsnu.band.example;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import androidx.appcompat.app.AppCompatActivity;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import org.mrsnu.band.BackendType;
import org.mrsnu.band.Band;
import org.mrsnu.band.Config;
import org.mrsnu.band.ConfigBuilder;
import org.mrsnu.band.CpuMaskFlag;
import org.mrsnu.band.Device;
import org.mrsnu.band.Engine;
import org.mrsnu.band.Model;
import org.mrsnu.band.Request;
import org.mrsnu.band.SchedulerType;
import org.mrsnu.band.SubgraphPreparationType;
import org.mrsnu.band.Tensor;

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
    b.addCPUMask(CpuMaskFlag.ALL);
    b.addPlannerCPUMask(CpuMaskFlag.PRIMARY);
    b.addWorkers(new Device[] {Device.CPU, Device.GPU, Device.DSP, Device.NPU});
    b.addWorkerNumThreads(new int[] {1, 1, 1, 1});
    b.addWorkerCPUMasks(
        new CpuMaskFlag[] {CpuMaskFlag.ALL, CpuMaskFlag.ALL, CpuMaskFlag.ALL, CpuMaskFlag.ALL});
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
      Log.i("BAND_JAVA",
          String.format(
              "Successfully created input (%d) / output (%d) tensors.", numInputs, numOutputs));

      float[] data = new float[2];
      data[0] = 1.f;
      data[1] = 2.f;
      ByteBuffer inputByteBuffer = ByteBuffer.allocateDirect(data.length * 4);
      inputByteBuffer.order(ByteOrder.nativeOrder());
      FloatBuffer inputFloatBuffer = inputByteBuffer.asFloatBuffer();
      inputFloatBuffer.put(data);
      inputTensors.get(0).setData(inputByteBuffer);

      List<Model> models = new ArrayList<>();
      models.add(model);
      List<List<Tensor>> inputTensorLists = new ArrayList<>();
      inputTensorLists.add(inputTensors);
      List<Request> reqs = engine.requestAsyncBatch(models, inputTensorLists);
      Log.i("BAND", "Requested!");
      engine.wait(reqs.get(0), outputTensors);

      ByteBuffer outputByteBuffer = outputTensors.get(0).getData();
      outputByteBuffer.order(ByteOrder.nativeOrder());
      int size = 1;
      String dim_string = "(";
      for (int dim : outputTensors.get(0).getDims()) {
        dim_string += String.format("%d, ", dim);
        size *= dim;
      }
      dim_string += ")";
      Log.i("RESULT", "output size: " + dim_string);

      for (int i = 0; i < size; i++) {
        Log.i(
            "RESULT", String.format("output[%d]: %f", i, outputByteBuffer.asFloatBuffer().get(i)));
      }
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath)
      throws IOException {
    AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }
}
