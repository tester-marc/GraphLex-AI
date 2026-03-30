# __main__.py: this enables "python -m app.ingestion" from terminal
#
# this file is the entry point when you run the package with
# "python -m app.ingestion --extract".
#
# __init__.py: runs on IMPORT (it sets up public API)
# __main__.py: runs on EXECUTE (starts CLI)
#
# The flow is as follows:
# 1. Python finds and runs this file here
# 2. main() from run.py parses the CLI args (--extract, --compare, --document)
# 3. main() routes to appropriate workflow (e.g., run_extract())
#

"""allows running as "python -m app.ingestion" """
from app.ingestion.run import main

# no "if __name__ == "__main__": " is needed here as this file is never imported
main()
