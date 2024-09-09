#!/bin/bash
jupyter notebook --no-browser --port=8080

# Run the following on local machine to connect to notebook, where <hostname> is the name of the host where this script is run
# Make sure ~/.ssh/config is configured such that it is possible to connect to this host using 'ssh <hostname>'
#ssh -L 8080:localhost:8080 <hostname>
