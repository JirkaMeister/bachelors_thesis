import os
import sys
import json
models = ["RNN", "GRU", "LSTM"]
#datasets = ["MNIST"]
dataset = "MNIST"
actions = ["Train", "Test"]
original_dir = os.getcwd()
result_file = "result.json"
iterations = 5

# def prepare_dir(dir):
#     os.chdir(dir)
#     sys.path.insert(0, os.getcwd())
#     if 'netModel' in sys.modules:
#         del sys.modules['netModel']

# def cleanup_dir():
#     os.chdir(original_dir)
#     if os.getcwd() in sys.path:
#         sys.path.remove(os.getcwd())

# def get_accuracy():
#     with open(result_file) as file:
#         result = json.load(file)
#         return result["accuracy"]

# with open(result_file, "w") as file:
#     pass

# for model in models:
#     sum_accuracy = 0
#     for i in range(iterations):
#         model_name = model
#         print(f"Model: {model_name}")
#         prepare_dir(f"{model}/{dataset}")

#         for action in actions:
#             script_name = f"net{action}.py"
#             with open(script_name) as file:
#                 script = file.read()
#             exec(script)

#         sum_accuracy += get_accuracy()

#         cleanup_dir()
    
#     accuracy = sum_accuracy / iterations

#     with open(result_file, "a") as file:
#         result = {"model": model_name, "accuracy": accuracy}
#         json.dump(result, file)
#         file.write("\n")


class AccuracyComparison:
    def __init__(self):
        with open(result_file, "w"):
            pass
        self.run()

    def run(self):
        for model in models:
            sum_accuracy = 0
            model_name = model
            print(f"Model: {model_name}")
            self.prepare_dir(f"{model}/{dataset}")

            for i in range(iterations):
                for action in actions:
                    script_name = f"net{action}.py"
                    with open(script_name) as file:
                        script = file.read()
                    exec(script)

                sum_accuracy += self.get_accuracy()

            self.cleanup_dir()
            
            self.write_result(sum_accuracy, model_name)

    def prepare_dir(self, dir):
        os.chdir(dir)
        sys.path.insert(0, os.getcwd())
        if 'netModel' in sys.modules:
            del sys.modules['netModel']

    def cleanup_dir(self):
        os.chdir(original_dir)
        if os.getcwd() in sys.path:
            sys.path.remove(os.getcwd())

    def get_accuracy(self):
        with open(result_file) as file:
            result = json.load(file)
            return result["accuracy"]
        
    def write_result(self, sum_accuracy, model_name):
        accuracy = sum_accuracy / iterations

        with open(result_file, "a") as file:
            result = {"model": model_name, "accuracy": accuracy, "samples": iterations}
            json.dump(result, file)
            file.write("\n")

AccuracyComparison()