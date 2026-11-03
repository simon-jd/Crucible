"""UNTRUSTED-side bootstrap, executed exclusively inside an isolation runtime.

No test expectations, success markers, or trusted grader live here. Candidate can
change this interpreter's state and forge its output; that output is only an
answer proposal to the independent host grader.
"""

import base64
import json
import sys


def main():
    request = json.loads(base64.b64decode(sys.argv[1], validate=True))
    namespace = {"__name__": "__candidate__"}
    exec(compile(request["solution"], "/solution.py", "exec"), namespace)  # noqa: S102 - guest only
    answer = namespace["solve"](request["input"])
    sys.stdout.write(json.dumps(answer, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
