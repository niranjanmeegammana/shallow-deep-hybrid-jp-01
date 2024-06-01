import sys

def get_args():
    # Get the number of command-line arguments
    n = len(sys.argv)
    if n > 1:
        # If there are command-line arguments, return the count excluding the script name
        return sys.argv[n - 1]
    else:
        # If no command-line arguments (except the script name itself), return a default value of 500
        return 500

print(get_args())
