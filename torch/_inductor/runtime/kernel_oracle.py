import itertools, math, pickle, requests, os
from copy import copy
import numpy as np
from pathlib import Path
from typing import Tuple
from pprint import pprint
import logging

def confLogger(logger, level):
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers = []
    formatter = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

log = logging.getLogger(__name__)

# confLogger(log, logging.DEBUG)
confLogger(log, logging.INFO)


def ceiling_log2(n):
    """Returns the power p such that 2^p is the smallest power of 2 >= n
    """
    if n <= 0:
        raise ValueError("Input must be positive")
    if n == 1:
        return 0
    
    # Count the number of bits needed
    return (n - 1).bit_length()


def nearest_pow2(n):
    return 2**ceiling_log2(n)


def sort_size_hints(d):
    def custom_sort_key(key):
        if key == 'x':
            return (0, 0)
        elif key == 'y':
            return (1, 0)
        elif key == 'z':
            return (2, 0)
        elif key.startswith('r'):
            # Extract number after 'r', handle 'r0_' format
            num_str = key[1:].rstrip('_')
            try:
                num = int(num_str)
                return (3, num)  # r keys sorted by number
            except ValueError:
                return (4, key)  # fallback for malformed r keys
        else:
            return (5, key)  # other keys at end
    return dict(sorted(d.items(), key=lambda item: custom_sort_key(item[0])))


class MockConfig:

    def __init__(self, cfg: dict, num_warps=1, num_stages=1):
        self.cfg = cfg
        self.num_warps = num_warps
        self.num_stages = num_stages

    def __repr__(self):
        return f"<Config {self.cfg}, warps={self.num_warps}, stages={self.num_stages}>"


class KernelOracle:
    # feel free to subclass by defining these class variables only:
    common_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] # WARNING: use powers of two
    """Note on the very small xblock values: for big numel values, these are filtered out we get grid overflow (see n_max_grids)
    Say we have numel=121..  we can still try blocksize 128 allright
    very small numels, i.e. < 1000 -> don't try small blocksizes, but directly the nearest upper pow2?

    Remember that each grid point has several threads (n warps = thread groups)
    """
    # GPU constraints
    # the idea here is: we should have a subclass per gfx
    # alternative: we can pass the device properties in the ctor
    n_max_grids = (2147483647, 65535, 65535)
    threads_per_warp = 64
    min_allowed_warps = 1 # per block
    max_allowed_warps = math.inf # per block # .. eh we already define min & max warps in num_warps_try
    max_threads_per_block = 1024
    num_warps_try = [1, 2, 4, 8]
    # num_stages_try = [1, 2] # dont ever use 4! -> crassh
    num_stages_try = [1] # this is fairly enough
    # parameters space ~ < 9000
    # model weights address:
    GITHUB_URL = 'https://github.com/AmdSampsa/kernelOracleWeights/raw/refs/heads/master/mi350_playground.pkl'
    # LOCAL_MODEL_PATH = "/tmp/mi350_playground.pkl"
    LOCAL_MODEL_PATH = Path(__file__).resolve().parent / Path("mi350_playground.pkl") # if we want to distribute with a pytorch branch
    configClass = MockConfig

    def __init__(self, path_to_model = None, fetch_model = True):
        """Create an instace of this as per a single "I have these size hints, what block values I should use" problem?
        """
        # self.size_hints = size_hints
        # self.original_size_hints = size_hints.copy()
        """
        kernels full decorators has this structure

            ::

                size_hints={'y': 33554432, 'x': 16}, # can also be "r_", "r0_", etc.
                filename=__file__,
                triton_meta={'signature':..,  'device': DeviceProperties(type='hip', ..., max_threads_per_multi_processor=2048, warp_size=64), 
                    'constants': {}, 'configs': }
                inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', ..}

        we use explicitly here just size_hints and kernel_name, but we could use more..

        """
        #self.kernel_name = kernel_name
        #self.nametag = 0 # a single number describing the flavour of this kernel. loaded from model metadata
        if (path_to_model is None) and fetch_model: # no path defined, user wants to use remote model data
            if not os.path.exists(self.LOCAL_MODEL_PATH):
                print(f'Model not found locally from {self.LOCAL_MODEL_PATH} Fetching from GitHub...')
                raise BaseException("KernelOracle: fetching from github disabled")
                if self.fetch_model_from_github():
                    self.path = self.LOCAL_MODEL_PATH
                else: 
                    self.path = None
            else:
                # print("Found local model file", self.LOCAL_MODEL_PATH)
                self.path = self.LOCAL_MODEL_PATH
        elif path_to_model: # user wants to use a local copy defined by him/herself
            self.path = path_to_model
        else: # no local model, dont fetch remote model (debugging or training mode)
            self.path = None
        
        if self.path is not None:
            self.__load_model_with_metadata__() # populates self.model and self.meta
            # self.nametag = self.generate_filetag(self.kernel_name)
        else:
            self.model = None # indicates we could not load the model or user just want to debug
            self.meta = {}

        """
        # oh.. all this is repeated/done independently in genConfigs2
        # this is the order we want:
        target_list = ["x", "y", "z", "r", "r0_", "r1_", "r2_", "r3_", "r4_", "r5_"]
        # for the ML model we want all dimensions
        self.size_hints_list=[1, 1, 1] # must be power of two
        self.types=[0, 0, 0]
        self.blockmap = [None, None, None] # give index -> spits out "XBLOCK", "YBLOCK", etc.
        i=0
        for key in target_list: # enforce key order of target_list
            if key in size_hints:
                self.size_hints_list[i] = size_hints[key] # i.e. size_hints[0] = xnumel, etc.
                if "r" in key:
                    self.types[i] = 2 # reduction dimension
                else:
                    self.types[i] = 1 # cartesian dimension
                self.blockmap[i] = key.upper() + "BLOCK" # i.e. XBLOCK, YBLOCK, R0_BLOCK, etc.
                i+=1
        """
        """
        print("KernelOracle: size_hints=", self.size_hints)
        print("KernelOracle: types=", self.types)
        print("KernelOracle: blockmap=", self.blockmap)
        """

    @classmethod
    def installPkgsIf(cls):
        try:
            import xgboost, sklearn, tabulate, pandas
        except ModuleNotFoundError as e:
            log.error("Modules missing, will install some: '{}'", e)
            os.system("pip install --no-input pandas scikit-learn xgboost tabulate")
        else:
            log.debug("all packages installed allright")

    @classmethod 
    def canDo(cls, size_hints: dict, kernel_name: str):
        """Check if we can use this optimizer for the kernel in question
        """
        if ("poi" in kernel_name) or ("red" in kernel_name): # pointwise and reduction ok
            return True
        return False


    @classmethod
    def tab_transform_to_powers_of_two(cls, df):
        # NOTE: we can have XBLOCK, YBLOCK, ZBLOCK, R0_BLOCK, R1_BLOCK, etc. whatever combo
        #
        # Create a copy of the DataFrame to avoid modifying the original one
        transformed_df = df.copy()
        # Transform each specified column to powers of 2
        # for col in df.columns.str.startswith("BLOCK") | df.columns.str.startswith("numel"):
        for col in ["BLOCK1", "BLOCK2", "BLOCK3", "numel1", "numel2", "numel3"]:
            transformed_df[col] = 2 ** transformed_df[col]
        return transformed_df


    def __bool__(self):
        return self.model is not None


    def __str__(self):
        st = f"<{self.__class__.__name__} / path: {str(self.path)}"
        if self.model is None:
            st += " / DEFUNCT"
        st += ">"
        return st


    # Function to download the model file from GitHub
    def fetch_model_from_github(self):
        """Fetch model binary from GitHub and save to local path."""
        # os.makedirs(os.path.dirname(self.LOCAL_MODEL_PATH), exist_ok=True)  # Ensure directories exist
        try:
            response = requests.get(self.GITHUB_URL)
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

            with open(self.LOCAL_MODEL_PATH, 'wb') as f:
                f.write(response.content)
            log.debug(f'Model downloaded and saved to {self.LOCAL_MODEL_PATH}')
            return True
        
        except requests.exceptions.HTTPError as http_err:
            log.error(f'HTTP error occurred: {http_err}')  # Log HTTP errors
        except requests.exceptions.ConnectionError as conn_err:
            log.error(f'Connection error occurred: {conn_err}')  # Log connection errors
        except requests.exceptions.Timeout as timeout_err:
            log.error(f'Timeout error occurred: {timeout_err}')  # Log timeout errors
        except Exception as err:
            log.error(f'An error occurred: {err}')  # Log any other errors

    def __load_model_with_metadata__(self):
        try:
            with open(self.path, 'rb') as file:
                loaded_model_with_metadata = pickle.load(file)
            # Access the model and metadata
            self.model = loaded_model_with_metadata["model"]
            self.meta = loaded_model_with_metadata["meta"]
        except Exception as e:
            log.error(f'Could not load the model, reason: {e}')
            self.model = None
            self.meta = {}


    def generate_filetag(self, filename):
        """kernel type to number mapping
        """
        mapper = self.meta["filename2tag"]
        for key in mapper:
            if key in filename:
                return mapper[key]
        return 0

    @classmethod
    def validate_config(cls, block_sizes: Tuple[int, ...], numel_sizes, num_warps: int) -> bool:
        """DEPRECATED
        
        Basic validation of a config
        
        block_sizes: (xblock, yblock, zblock)
        numel_sizes: (xnumel, ynumel, znumel)
        """
        # TODO: not really sure how to limit total number of threads
        # .. as triton doesn't deal with threads at all, but just with the number of parallelization
        # log.debug('validate config block-sizes: %s warps: %s', block_sizes, num_warps)
        log.debug('validate config: (%i, %i, %i, %i)', block_sizes[0], block_sizes[1], block_sizes[2], num_warps)
        min_allowed_threads  = cls.min_allowed_warps*cls.threads_per_warp
        max_allowed_threads = cls.max_allowed_warps*cls.threads_per_warp
        total_elements = 1 # total number of elements in a block
        for block_size in block_sizes:
            total_elements *= block_size # NOTE: blocksize = 1 has no effect
        # basic sanity check: blocksize <= numel
        # blocksize can be numel size rounded to the nearest pow of two
        for block_size, numel_size in zip(block_sizes, numel_sizes):
            if block_size > nearest_pow2(numel_size):
                log.debug('skip: block_size > nearest_pow2(numel_size)')
                return False
        # Basic thread count checks
        if total_elements > max_allowed_threads:
            log.debug('skip: max threads exceeded')
            return False
        # TODO what if we have just 32 elements?
        if total_elements < min_allowed_threads:
            log.debug('skip: min threads underflow')
            return False

        threads_per_block = num_warps * cls.threads_per_warp  # 64 threads per warp for AMD
        if threads_per_block > cls.max_threads_per_block:
            log.debug('skip: threads per block %s overflow', threads_per_block)
            return False

        # TODO what if we have just 32 elements?
        if total_elements < threads_per_block:
            # 1D example: say, numel_size = 1024, total_elements/block_size = 1 -> grid=1024, i.e. one grid point per element
            # 64 threads (one warp) per one grid point per one elemetnt -> crazy
            log.debug('skip: more threads than elements')
            return False

        # Warp alignment
        if total_elements % num_warps != 0:
            log.debug('skip: warps not aligned')
            return False
        # Work per warp
        # there's num_warps warps per block.  do the elements/threads distribute equally into each warp?
        work_per_warp = total_elements // num_warps
        if work_per_warp < cls.threads_per_warp:
            log.debug('skip: bad work distribution among warps')
            return False
        # Grid size checks
        i=0
        for block_size, numel_size in zip(block_sizes, numel_sizes):
            grid_size = (numel_size + block_size - 1) // block_size # (1+1-1)//1 = 1 -> i.e. numel=1 & block_size=1 -> no effect
            if grid_size > cls.n_max_grids[i]:
                log.debug('max grid size exceeded: numel_size %s, block_size %s, grid_size %s', numel_size, block_size, grid_size)
                return False
            i+=1
        log.debug(f'config OK. elements-per-grid-point={total_elements}, threads-per-grid-point={threads_per_block}')
        return True


    @classmethod
    def genConfigs(cls, xnumel=1, ynumel=1, znumel=1, nametag=0):
        """DEPRECATED
        
        Generate a set of configurations and return them as pandas dataframe

        Generates configs for "asking" from the trained model the best xblock, etc. values
        the numel values are given as size_hints in inductor, i.e. they are always the nearest pow of two
        """
        import pandas as pd
        numel_sizes = (xnumel, ynumel, znumel)
        # normalize
        xnumel_log2 = np.log2(xnumel)
        ynumel_log2 = np.log2(ynumel)
        znumel_log2 = np.log2(znumel)
        data = []
        # Loop through the different values for XBLOCK, YBLOCK, num_warps, and num_stages
        for XBLOCK in cls.common_sizes:
            for YBLOCK in cls.common_sizes:
                for ZBLOCK in cls.common_sizes:
                    for num_warps in cls.num_warps_try:
                        for num_stages in cls.num_stages_try:
                            if cls.validate_config(((XBLOCK, YBLOCK, ZBLOCK)), numel_sizes, num_warps):
                                # Append the current configuration as a dictionary to the data list
                                data.append({
                                    'name': nametag,
                                    'XBLOCK': np.log2(XBLOCK),    
                                    'YBLOCK': np.log2(YBLOCK),    
                                    'ZBLOCK': np.log2(ZBLOCK),    
                                    'num_warps': num_warps,  
                                    'num_stages': num_stages, 
                                    'xnumel': xnumel_log2,    
                                    'ynumel': ynumel_log2,   
                                    'znumel': znumel_log2     
                                })
        # Convert the list of dictionaries into a DataFrame
        return pd.DataFrame(data)


    @classmethod
    def genConfigs2(cls, size_hints: dict, nametag=0):
        """Based on inductor size_hints, generate a list of blocksize combinations.  Values are normalized
        to powers of 2.
        
        Like genConfigs but does not assume cartesian block dimensions (XBLOCK, YBLOCK, ZBLOCK), but
        the blocks can be any combination of cartesian and reduction blocks, say (XBLOCK, R0_BLOCK)
        or (XBLOCK, YBLOCK, R0_BLOCK), etc.
        
        Generate a set of configurations and return them as pandas dataframe

        Generates configs for "asking" from the trained model the best xblock, etc. values
        the numel values are given as size_hints in inductor, i.e. they are always the nearest pow of two
        """
        import pandas as pd
        size_hints = sort_size_hints(size_hints) # sort into x,y,r,r0_,r1_,etc. order
        numels = [1, 1, 1] # must be powers of 2
        types = [0, 0, 0] # 0 = void, 1 = cartesian, 2 = reduction
        block_sizes = [[1], [1], [1]]
        blockmap = [None, None, None] # give index -> spits out "XBLOCK", "YBLOCK", etc.
        numelmap = [None, None, None] # give index -> spits out "xnumel", "ynumel", etc.
        i=0
        for key, value in size_hints.items():
            blockname = key + "block" # i.e. r0_block, xblock, etc
            blockname = blockname.upper() # R0_BLOCK, XBLOCK, etc.
            if "R" in blockname:
                types[i] = 2 # reduction
            else:
                types[i] = 1 # i.e. X/Y/Z
            numels[i] = value
            block_sizes[i] = cls.common_sizes
            blockmap[i] = key.upper() + "BLOCK" # i.e. XBLOCK, YBLOCK, R0_BLOCK, etc.
            numelmap[i] = key + "numel" # i.e. xnumel, ynumel, r0_numel, etc.
            i+=1
        """
        print("genConfigs2> numels=", numels)
        print("genConfigs2> types=", types)
        print("genConfigs2> block_sizes=", block_sizes)
        """
        data = []
        for BLOCK1 in block_sizes[0]:
            for BLOCK2 in block_sizes[1]:
                for BLOCK3 in block_sizes[2]:
                    for num_warps in cls.num_warps_try:
                            for num_stages in cls.num_stages_try:
                                if cls.validate_config(((BLOCK1, BLOCK2, BLOCK3)), numels, num_warps):
                                    data.append({
                                            'name': nametag,
                                            'numel1': np.log2(numels[0]),
                                            'numel2': np.log2(numels[1]),
                                            'numel3': np.log2(numels[2]),
                                            'BLOCK1': np.log2(BLOCK1),    
                                            'BLOCK2': np.log2(BLOCK2),    
                                            'BLOCK3': np.log2(BLOCK3),
                                            'type1': types[0], # 0 = nada, 1 = cartesian, 2 = reduction
                                            'type2': types[1],
                                            'type3': types[2],
                                            'num_warps': num_warps,  
                                            'num_stages': num_stages
                                        })
        return pd.DataFrame(data)

    
    def rankConfigs(self, size_hints: dict, name: str, nmax):
        """Generate configs and rank them using the ML regressor.  Return the best configs.
        """
        # here we go again..
        size_hints_sorted = sort_size_hints(size_hints)
        # print("sorted size_hints>", size_hints_sorted)
        blockmap = [None, None, None] # give index -> spits out "XBLOCK", "YBLOCK", etc.
        i=0
        for key, value in size_hints_sorted.items():
            blockname = key + "block" # i.e. r0_block, xblock, etc
            blockname = blockname.upper() # R0_BLOCK, XBLOCK, etc.
            blockmap[i] = key.upper() + "BLOCK" # i.e. XBLOCK, YBLOCK, R0_BLOCK, etc.
            i+=1

        input_df = self.genConfigs2(
            size_hints_sorted,
            nametag= self.generate_filetag(name)
        ).astype(float) # NOTE: float

        # run prediction on all configurations:
        if self.model is None:
            log.error("Model not defined, returning empty list of configs")
            return []
        predictions = self.model.predict(input_df)
        # add predictions to the table
        comb = input_df.copy()
        comb["Y"] = predictions
        # sort: smaller the better
        #print(type(comb))  # Should be <class 'pandas.core.frame.DataFrame'>
        #print(comb.columns)  # Check if 'Y' column exists
        sorted_ = comb.sort_values(by='Y', ascending=True)
        # return sorted_
        final=self.tab_transform_to_powers_of_two(sorted_)
        # final = sorted_
        configs = []

        # finally, we need to go from BLOCK1, BLOCK2, etc. to XBLOCK, YBLOCK, R0_BLOCK, etc.
        for i, row  in enumerate(final.iterrows()):
            data = row[1] # row[0] is the index
            # print(">>", data)
            cfg = {}
            # self.blockmap gives correct names of the different blocks
            # i.e. XBLOCK, YBLOCK, etc.
            # data["type1"] tells us if there is something in BLOCK1 or not
            if data["type1"] > 0: # 1 = cartesian, 2 = reduction
                cfg[blockmap[0]] = int(data["BLOCK1"])
            if data["type2"] > 0:
                cfg[blockmap[1]] = int(data["BLOCK2"])
            if data["type3"] > 0:
                cfg[blockmap[2]] = int(data["BLOCK3"])
            # so now we have cfg["XBLOCK"] etc.
            Config = self.configClass
            configs.append(Config(cfg, num_warps=int(data["num_warps"]), num_stages=int(data["num_stages"])))
            if i>=nmax: # some reasonable cutoff..
                break
        return configs

    def getBest(self, size_hints: dict, name: str):
        return self.rankConfigs(size_hints, name, nmax=1)[0]


    def train(self, csv_file: Path):
        from tabulate import tabulate
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        import pandas as pd
        from sklearn.metrics import mean_squared_error
        from paas.downstream import filename2tag

        print("reading file", csv_file)
        df = pd.read_csv(csv_file).astype(float)
        # Define features and target variable
        X = df.drop(columns=['Y'])
        y = df['Y']
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Initialize the XGBoost regressor
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        print("fitting model")
        model.fit(X_train, y_train)
        print("testing model")
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse:.6f}')
        model_filename = self.LOCAL_MODEL_PATH
        # Combine model and metadata
        model_with_metadata = {
            "model": model,
            "meta": {
                "filename2tag": filename2tag
            }
        }
        # Save the combined model and metadata using pickle
        with open(model_filename, 'wb') as file:
            pickle.dump(model_with_metadata, file)
        print(f'Model and metadata saved to {model_filename}')


def train():
    import argparse
    parser = argparse.ArgumentParser(description="Train the KernelOracle")
    parser.add_argument('--data', type=str, required=True, help='Path to training data csv file')
    p = parser.parse_args()
    ko = KernelOracle(path_to_model = None, fetch_model = True)
    ko.train(p.data)

