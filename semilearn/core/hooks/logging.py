from .hook import Hook


class LoggingHook(Hook):
    def __init__(self) -> None:
        super().__init__()

    def after_train_step(self, algorithm):
        """must be called after evaluation"""
        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            if not algorithm.distributed or (algorithm.distributed
                                             and algorithm.rank %
                                             algorithm.ngpus_per_node == 0):
                algorithm.print_fn(
                    f"-----------------------{algorithm.it + 1} iteration: Validating Start!------------------------"
                )
                algorithm.print_fn(
                    f"Closed-Set Evaluation: Acc:{algorithm.eval_dict['c_acc'] * 100:.2f}, Precision:{algorithm.eval_dict['c_precision'] * 100:.2f}, RECALL:{algorithm.eval_dict['c_recall'] * 100:.2f}, F1-score:{algorithm.eval_dict['c_f1'] * 100:.2f}"
                )
                algorithm.print_fn(
                    f"Open-Set Evaluation: Acc:{algorithm.eval_dict['o_acc'] * 100:.2f}, Precision:{algorithm.eval_dict['o_precision'] * 100:.2f}, RECALL:{algorithm.eval_dict['o_recall'] * 100:.2f}, F1-score:{algorithm.eval_dict['o_f1'] * 100:.2f}"
                )
                algorithm.print_fn(
                    f"BEST_CLOSE_ACC: {algorithm.best_eval_acc * 100:.2f}, at {algorithm.best_it + 1} iters"
                )
                algorithm.print_fn(
                    f"-----------------------{algorithm.it + 1} iteration: Validating End!------------------------"
                )

            if algorithm.tb_log is not None:
                algorithm.tb_log.update(algorithm.tb_dict, algorithm.it)

        elif self.every_n_iters(algorithm, algorithm.num_log_iter):
            if not algorithm.distributed or (algorithm.distributed
                                             and algorithm.rank %
                                             algorithm.ngpus_per_node == 0):
                algorithm.print_fn(
                    f"{algorithm.it + 1} iteration, USE_EMA: {algorithm.ema_m != 0}, {algorithm.tb_dict}"
                )

    def after_run(self, algorithm):
        best_results = algorithm.best_results
        algorithm.print_fn("--------------------------Final Evaluation---------------------")
        algorithm.print_fn(f"Closed-Set Evaluation: Acc:{best_results['c_acc'] * 100:.2f}, Precision:{best_results['c_precision'] * 100:.2f}, RECALL:{best_results['c_recall'] * 100:.2f}, F1-score:{best_results['c_f1'] * 100:.2f}")
        algorithm.print_fn(f"Open-Set Evaluation: Acc:{best_results['o_acc'] * 100:.2f}, Precision:{best_results['o_precision'] * 100:.2f}, RECALL:{best_results['o_recall'] * 100:.2f}, F1-score:{best_results['o_f1'] * 100:.2f}")
