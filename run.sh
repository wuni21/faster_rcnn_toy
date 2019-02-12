sudo nvidia-docker run -it --rm -v $(pwd):$(pwd) -v /home/siit/data:/home/siit/data -v /home/siit/checkpoint:/home/siit/checkpoint -w $(pwd) \
                       acc3597/jaesung:pytorch_1.0 \
                       sh script.sh \

           
