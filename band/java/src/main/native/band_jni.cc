// Copyright 2023 Seoul National University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <jni.h>

#include "band/logger.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

JNIEXPORT JNICALL void Java_org_mrsnu_band_Band_nativeDoNothing(
    JNIEnv* env, jclass /*clazz*/) {
  // Do nothing. Used for check if the native library is loaded.
}

JNIEXPORT JNICALL void Java_org_mrsnu_band_Band_nativeSetVerbosity(
    JNIEnv* env, jclass /*clazz*/, jint verbosity) {
  band::Logger::Get().SetVerbosity(static_cast<band::LogSeverity>(verbosity));
}

JNIEXPORT JNICALL jstring
Java_org_mrsnu_band_Band_nativeGetLastLog(JNIEnv* env, jclass /*clazz*/) {
  return env->NewStringUTF(band::Logger::Get().GetLastLog().second.c_str());
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus