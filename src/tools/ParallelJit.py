
from numba import jit

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


def distribute(f, inds, f_args, f_kwargs={}, watcher=None, w_args=[], n_workers=8, timeout=1):
    '''
    Distribute a set of indices over a function that loops over indices.
    Indices are divided amongst multiple of these.

    The first argument of the function must be for a set of indices
    '''
    jobs = dict()

    print("Quick dry run..")
    f(inds[0:100], *f_args)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        print("Distributing..")
        for n in range(n_workers):
            sub_inds = inds[n::n_workers]
            job = executor.submit(f, sub_inds, *f_args)
            jobs[job] = sub_inds

        # Wait for jobs
        print("Waiting...")
        while len(jobs) > 0:
            done, notdone = concurrent.futures.wait(jobs, timeout=timeout)
            for job in done:
                print("\nJob done:", job)
                sub_inds = jobs.pop(job)
                print("Sub:", sub_inds[0])

                ex = job.exception()
                if ex is not None:
                    raise ex

            if watcher is not None:
                watcher(False, *w_args)

        if watcher is not None:
            watcher(True, *w_args)

        print("Jobs done")
