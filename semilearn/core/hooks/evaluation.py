import os
from .hook import Hook


# class EvaluationHook(Hook):
#     def __init__(self) -> None:
#         super().__init__()

#     def before_run(self, algorithm):
#         return
#         # algorithm.evaluate('eval')

#     def after_train_step(self, algorithm):
#         if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
#             algorithm.print_fn("\n-----------------------Validating start!------------------------")
#             eval_dict = algorithm.evaluate('eval')
#             algorithm.eval_dict.update(eval_dict)

#             # update best metrics
#             if algorithm.eval_dict['eval/top-1-acc'] > algorithm.best_eval_acc:
#                 algorithm.best_eval_acc = algorithm.eval_dict['eval/top-1-acc']
#                 algorithm.best_it = algorithm.it

#     def after_run(self, algorithm):
#         return

class EvaluationHook(Hook):
    def __init__(self) -> None:
        super().__init__()

    def before_run(self, algorithm):
        return
        # algorithm.evaluate('eval')

    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            results = algorithm.evaluate_open()
            algorithm.eval_dict.update(results)
            
            # update best metrics
            if results['c_acc'] >= algorithm.best_eval_acc:
                algorithm.best_eval_acc = results['c_acc']
                algorithm.best_it = algorithm.it
                algorithm.best_results = results

    def after_run(self, algorithm):
        return
