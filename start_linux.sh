
set -e

# Check if virtual environment exists, if not create it
if [ ! -d "env" ]; then
    python3 -m venv env
    python3 -m pip install -r requirements.txt
fi

# Activate virtual environment
source env/bin/activate

# Check if requirements are met, if not prompt user install, continue, or exit
# Yeah, I didn't implement this right.
# if ! python3 -m pip freeze | grep -q -F -f<<<"$(sed -n "s/\s*#.*$//;s/\s+//;/^$/d;p" requirements.txt)"; then
#     echo "Requirements not met, would you like to install them? (y/n)"
#     read response
#     if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
#         python3 -m pip install -r requirements.txt
#     else
#         echo "Would you like to launch the server anyway? (y/n)"
#         read response
#         if [[ ! "$response" =~ ^([nN][oO]|[nN][aA][hH]|[nN])+$ ]]; then
#             exit 1
#         fi
#     fi
# fi

/usr/bin/env python3 -m wordpllay "$@"