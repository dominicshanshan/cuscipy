## Requirement
    cupy version >9.x
    cmake version >3.15
    CUDA_HOME environment variable is properly set
## Docker for development environment
To build the docker image, go to the `docker` directory and run 
```bash
bash build.sh
``` 
To launch the environment, under the 'custats' root directory, run
```bash
docker run -it --rm --gpus all -v $PWD:/workspace custat
```
## Unit tests
In the development environment, run the unit test scripts
```bash
bash run_xxx_unit_tests.sh
```
