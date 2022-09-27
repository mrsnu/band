#ifndef BAND_BACKEND_TFL_BACKEND_H
#define BAND_BACKEND_TFL_BACKEND_H

#include "band/backend/tfl/interpreter.h"
#include "band/backend/tfl/model.h"
#include "band/backend/tfl/tensor.h"
#include "band/backend/tfl/util.h"
#include "band/interface/backend.h"
#include "band/interface/backend_factory.h"

namespace Band {
using namespace Interface;
namespace TfLite {
class InterpreterCreator : public Creator<IInterpreter> {
 public:
  IInterpreter* Create() const override { return new TfLiteInterpreter(); }
};

class ModelCreator : public Creator<IModel, ModelId> {
 public:
  IModel* Create(ModelId id) const override { return new TfLiteModel(id); }
};

class UtilCreator : public Creator<IBackendUtil> {
 public:
  IBackendUtil* Create() const override { return new TfLiteUtil(); }
};

}  // namespace TfLite
}  // namespace Band

#endif