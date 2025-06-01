import logging
from functools import wraps


LOGGER = logging.getLogger(__name__)


class GuideHandler:

    def __init__(self, ordered_steps):
        self.cur_iteration = 0
        self.current_step = -1
        self.start_point = -1
        self.ordered_steps = ordered_steps
        self.set_methods = set()

        self.producer_to_step_map = {}
        self.getter_to_step_map = {}

        self.iterations = []
        for idx, (keys, sets, gets) in enumerate(self.ordered_steps):
            self.iterations.append(-1)

            for prod in keys:
                self.producer_to_step_map[prod.__name__] = idx
            for prod in sets:
                self.producer_to_step_map[prod.__name__] = idx
                self.set_methods.add(prod.__name__)

            for get in gets:
                self.getter_to_step_map[get.__name__] = idx

    def or_join(self, methods):
        return " or ".join([method.__name__ for method in methods])

    def get_get_steps_in_between(self, cur_step, next_step):
        step_strs = []
        for step in range(cur_step + 1, next_step):
            step_strs.append(
                f"{step} {self.or_join(self.ordered_steps[step][2])}")
        return step_strs

    def get_last_up_to_date(self, next_step):
        latest_up_to_date = 0
        for step in range(next_step):
            if self.iterations[step] == self.cur_iteration:
                latest_up_to_date = step
        return latest_up_to_date

    def join_steps(self, step_strs):
        return "\n\t".join(step_strs)

    def get_steps_in_between(self, cur_step, next_step):
        step_strs = []
        for step in range(cur_step + 1, next_step):
            option_strs = []
            option_strs.extend(self.ordered_steps[step][0])
            option_strs.extend(self.ordered_steps[step][1])
            step_strs.append(f"{step}. {self.or_join(option_strs)}")
        return step_strs

    def log_next_producer_step(self, name):
        next_step = self.current_step + 1

        if next_step >= len(self.ordered_steps):
            cur_step_name = self.or_join(self.ordered_steps[self.current_step][0])
            LOGGER.warning((f"[GUIDE] DONE: {name}.\n"
                            f"\tYou have reached the end of the "
                            f"predictive engineering workflow.\n"
                            f"\tYou can call {cur_step_name} again or re-perform previous steps "
                            f"based on results."))
        else:
            next_step_name = self.or_join(self.ordered_steps[next_step][0])
            LOGGER.warning(f"[GUIDE] DONE: {name}.\n"
                           f"\tYou can perform the next step by calling {next_step_name}.")

    def perform_producer_step(self, zephyr, method,
                              *method_args, **method_kwargs):
        step_num = self.producer_to_step_map[method.__name__]
        res = method(zephyr, *method_args, **method_kwargs)
        self.current_step = step_num
        self.iterations[step_num] = self.cur_iteration
        self.log_next_producer_step(method.__name__)
        return res

    def try_log_forward_set_method_warning(self, name, next_step):
        if self.current_step != -1:
            from_str = (f"Going from step {self.current_step} to "
                        f"step {next_step} by performing {name}.")
        else:
            from_str = (f"Performing step {next_step} with {name}.")
        LOGGER.warning((f"[GUIDE] STALE WARNING: {name}.\n"
                        f"\t{from_str}\n"
                        f"\tThis is a forward step via a set method.\n"
                        f"\tAll previous steps' results will be considered stale."))

    def try_log_backwards_set_method_warning(self, name, next_step):
        LOGGER.warning((f"[GUIDE] STALE WARNING: {name}.\n"
                        f"\tGoing from step {self.current_step} to "
                        f"step {next_step} by performing {name}.\n"
                        f"\tThis is a backwards step via a set method.\n"
                        f"\tAll other steps' results will be considered stale."))

    def try_log_backwards_key_method_warning(self, name, next_step):
        steps_in_between = self.get_steps_in_between(next_step, self.current_step + 1)
        if len(steps_in_between) > 0:
            steps_in_between_str = (f"\tAny results produced by the following steps "
                                    f"will be considered stale:\n"
                                    f"\t{self.join_steps(steps_in_between)}")
        else:
            steps_in_between_str = ""

        LOGGER.warning((f"[GUIDE] STALE WARNING: {name}.\n"
                        f"\tGoing from step {self.current_step} to "
                        f"step {next_step} by performing {name}.\n"
                        f"\tThis is a backwards step via a key method.\n"
                        f"{steps_in_between_str}"))

    def log_get_inconsistent_warning(self, name, next_step):
        prod_steps_str = self.or_join(self.ordered_steps[next_step][0])
        prod_steps = f"{next_step}.{prod_steps_str}"
        latest_up_to_date = self.get_last_up_to_date(next_step)
        LOGGER.warning((f"[GUIDE] INCONSISTENCY WARNING: {name}.\n"
                        f"Unable to perform {name} because"
                        f"{prod_steps} has not been run yet.\n"
                        f"Run steps starting at or before {latest_up_to_date}."))

    def log_get_stale_warning(self, name, next_step):
        latest_up_to_date = self.get_last_up_to_date(next_step)
        LOGGER.warning((f"[GUIDE] STALE WARNING: {name}.\n"
                        f"This data is potentially stale.\n"
                        f"Re-run steps starting at or before {latest_up_to_date}"
                        f"to ensure data is up to date."))

    # tries to perform step if possible -> warns that data might be stale

    def try_perform_forward_producer_step(
            self, zephyr, method, *method_args, **method_kwargs):
        name = method.__name__
        next_step = self.producer_to_step_map[name]
        if name in self.set_methods:  # set method will update start point and start new iteration
            self.try_log_forward_set_method_warning(name, next_step)
            self.start_point = next_step
            self.cur_iteration += 1
        # next_step == 0, set method (already warned), or previous step is up
        # to term
        res = self.perform_producer_step(
            zephyr, method, *method_args, **method_kwargs)
        return res

    def try_perform_backward_producer_step(
            self, zephyr, method, *method_args, **method_kwargs):
        name = method.__name__
        next_step = self.producer_to_step_map[name]
        # starting new iteration
        self.cur_iteration += 1
        if next_step == 0 or name in self.set_methods:
            self.start_point = next_step
        else:  # key method
            # mark everything from start point to next step as current term
            for i in range(self.start_point, next_step):
                if self.iterations[i] != -1:
                    self.iterations[i] = self.cur_iteration

        if name in self.set_methods:
            self.try_log_backwards_set_method_warning(name, next_step)
        else:
            self.try_log_backwards_key_method_warning(name, next_step)

        res = self.perform_producer_step(
            zephyr, method, *method_args, **method_kwargs)

        return res

    def try_perform_producer_step(
            self, zephyr, method, *method_args, **method_kwargs):
        name = method.__name__
        next_step = self.producer_to_step_map[name]
        if next_step >= self.current_step:
            res = self.try_perform_forward_producer_step(
                zephyr, method, *method_args, **method_kwargs)
            return res
        else:
            res = self.try_perform_backward_producer_step(
                zephyr, method, *method_args, **method_kwargs)
            return res

    # dont update current step or terms

    def try_perform_inconsistent_producer_step(  # add using stale and overwriting
            self, zephyr, method, *method_args, **method_kwargs):
        name = method.__name__
        next_step = self.producer_to_step_map[name]
        # inconsistent forward step: performing key method but previous step is
        # not up to date
        if (next_step >= self.current_step and
                self.iterations[next_step - 1] != self.cur_iteration):
            prev_step = next_step - 1
            prev_set_method = self.or_join(self.ordered_steps[prev_step][1])
            prev_key_method = self.or_join(self.ordered_steps[prev_step][0])
            if next_step == len(self.ordered_steps) - 1:
                final_text = (f"\tOtherwise, you can regenerate the data of the previous "
                              f"step by calling {prev_key_method}, and then call {name} again.")
            else:
                corr_set_method = self.or_join(self.ordered_steps[next_step][1])
                final_text = (f"\tIf you already have the data for THIS step, you can use "
                              f"{corr_set_method} to set the data.\n"
                              f"\tOtherwise, you can regenerate the data of the "
                              f"previous step by calling {prev_key_method}, "
                              f"and then call {name} again.")
            LOGGER.warning(f"[GUIDE] INCONSISTENCY WARNING: {name}\n"
                           f"\tUnable to perform {name} because you are "
                           f"performing a key method at step {next_step} but the result of the "
                           f"previous step, step {prev_step}, is stale.\n"
                           f"\tIf you want to use the stale result or "
                           f"already have the data for step {prev_step}, you can use "
                           f"{prev_set_method} to set the data.\n"
                           f"{final_text}")
        elif (next_step < self.current_step and
              self.iterations[next_step - 1] != self.cur_iteration):
            prev_step = next_step - 1
            prev_key_method = self.or_join(self.ordered_steps[prev_step][0])
            prev_set_method = self.or_join(self.ordered_steps[prev_step][1])

            if next_step == len(self.ordered_steps) - 1:
                final_text = (f"\tOtherwise, you can regenerate the data of the previous "
                              f"step by calling {prev_key_method}, and then call {name} again.")
            else:
                corr_set_method = self.or_join(self.ordered_steps[next_step][1])
                final_text = (f"\tIf you already have the data for THIS step, you can use "
                              f"{corr_set_method} to set the data.\n"
                              f"\tOtherwise, you can regenerate the data of the "
                              f"previous step by calling {prev_key_method}, "
                              f"and then call {name} again.")
            LOGGER.warning(f"[GUIDE] INCONSISTENCY WARNING: {name}\n"
                           f"\tUnable to perform {name} because "
                           f"you are going backwards and starting a new iteration by "
                           f"performing a key method at step {next_step} but the result of the "
                           f"previous step, step {prev_step}, is STALE.\n"
                           f"\tIf you want to use the STALE result or "
                           f"already have the data for step {prev_step}, you can use "
                           f"{prev_set_method} to set the data.\n"
                           f"{final_text}")

    def try_perform_getter_step(
            self, zephyr, method, *method_args, **method_kwargs):
        name = method.__name__
        # either inconsistent, stale, or up to date
        step_num = self.getter_to_step_map[name]
        step_iteration = self.iterations[step_num]
        if step_iteration == -1:
            self.log_get_inconsistent_warning(name, step_num)
        elif step_iteration == self.cur_iteration:
            res = method(zephyr, *method_args, **method_kwargs)
            return res
        else:
            self.log_get_stale_warning(name, step_num)
            res = method(zephyr, *method_args, **method_kwargs)
            return res

    def guide_step(self, zephyr, method, *method_args, **method_kwargs):
        method_name = method.__name__
        if method_name in self.producer_to_step_map:
            # up-todate
            next_step = self.producer_to_step_map[method_name]
            if (next_step == 0 or  # 0 step always valid, starting new iteration
                # set method always valid, but will update start point and
                # start new iteration
                method_name in self.set_methods or
                    # key method valid if previous step is up to date
                    self.iterations[next_step - 1] == self.cur_iteration):
                # forward step only valid if set method or key method w/ no
                # skips
                res = self.try_perform_producer_step(
                    zephyr, method, *method_args, **method_kwargs)
                return res
            else:  # stale or inconsistent
                res = self.try_perform_inconsistent_producer_step(
                    zephyr, method, *method_args, **method_kwargs)
                return res
        elif method_name in self.getter_to_step_map:
            res = self.try_perform_getter_step(
                zephyr, method, *method_args, **method_kwargs)
            return res
        else:
            print(f"Method {method_name} does not need to be wrapped")


def guide(method):

    @wraps(method)
    def guided_step(instance, *method_args, **method_kwargs):
        return instance._guide_handler.guide_step(
            instance, method, *method_args, **method_kwargs)

    return guided_step
