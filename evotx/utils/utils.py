import os
import re
import json


def check_cache(cache_dir, keyword):
    keyword = keyword.lower()
    if os.path.exists(os.path.join(cache_dir, f"{keyword}.json")):
        return json.load(open(os.path.join(cache_dir, f"{keyword}.json"), 'r'))
    return None


def get_transaction_hash(txn_link):
    pattern = r'(0x[0-9a-fA-F]{64})'
    return re.findall(pattern, txn_link)


# =====================================================================
#                    Chain Normalization
# =====================================================================

def normalize_chain_name(chain: str) -> str:
    chain = (chain or "").strip().lower()
    alias = {
        "mainnet": "eth",
        "ethereum": "eth",
        "eth": "eth",
        "bsc": "bsc",
        "polygon": "polygon",
        "matic": "polygon",
        "avax": "avax",
        "avalanche": "avax",
        "opt": "opt",
        "optimism": "opt",
        "arbi": "arbi",
        "arbitrum": "arbi",
        "arb": "arbi",
        "op": "opt",
        "fantom": "fantom",
        "base": "base",
        "pulsechain": "pulsechain",
        "gnosis": "gnosis",
        "mantle": "mantle",
        "linea": "linea",
        "chain_59144": "linea",
        "chain_5000": "mantle",
        "chain_100": "gnosis",
    }
    return alias.get(chain, chain)


# =====================================================================
#                    Common Address Registry
# =====================================================================

COMMON_ADDRESS_REGISTRY = {
    "eth": {
        "system": [
            {
                "address": "0x0000000000000000000000000000000000000000",
                "name": "Zero Address",
                "tags": ["system", "null", "benign_baseline"]
            },
            {
                "address": "0x000000000000000000000000000000000000dEaD",
                "name": "Dead Address",
                "tags": ["system", "burn", "benign_baseline"]
            },
        ],
        "wrapped_native": [
            {
                "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
                "name": "WETH",
                "tags": ["wrapped_native", "token", "benign_baseline"]
            }
        ],
        "stablecoins": [
            {
                "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                "name": "USDC",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                "name": "USDT",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0x4Fabb145d64652a948d72533023f6E7A623C7C53",
                "name": "BUSD",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                "name": "DAI",
                "tags": ["stablecoin", "token", "benign_baseline"]
            }
        ],
        "bluechip_tokens": [
            {
                "address": "0xB8c77482e45F1F44dE1745F52C74426C631bDD52",
                "name": "BNB",
                "tags": ["bluechip", "token", "benign_baseline"]
            },
            {
                "address": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
                "name": "WBTC",
                "tags": ["bluechip", "token", "benign_baseline"]
            }
        ],
        "dex_routers": [
            {
                "address": "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                "name": "UniswapV2 Router",
                "tags": ["dex_router", "router", "benign_baseline"]
            }
        ],
        "dex_factories": [
            {
                "address": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
                "name": "UniswapV2 Factory",
                "tags": ["dex_factory", "factory", "benign_baseline"]
            }
        ],
        "bridges": [],
        "lending_core": [],
        "oracles": [],
        "vaults": [],
    },

    "bsc": {
        "system": [
            {
                "address": "0x0000000000000000000000000000000000000000",
                "name": "Zero Address",
                "tags": ["system", "null", "benign_baseline"]
            },
            {
                "address": "0x000000000000000000000000000000000000dEaD",
                "name": "Dead Address",
                "tags": ["system", "burn", "benign_baseline"]
            },
        ],
        "wrapped_native": [
            {
                "address": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
                "name": "WBNB",
                "tags": ["wrapped_native", "token", "benign_baseline"]
            }
        ],
        "stablecoins": [
            {
                "address": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
                "name": "USDC",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0x55d398326f99059fF775485246999027B3197955",
                "name": "USDT",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56",
                "name": "BUSD",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0x1AF3F329e8BE154074D8769D1FFa4eE058B1DBc3",
                "name": "DAI",
                "tags": ["stablecoin", "token", "benign_baseline"]
            }
        ],
        "bluechip_tokens": [
            {
                "address": "0x2170Ed0880ac9A755fd29B2688956BD959F933F8",
                "name": "ETH (BSC)",
                "tags": ["bluechip", "token", "benign_baseline"]
            }
        ],
        "dex_routers": [
            {
                "address": "0x10ED43C718714eb63d5aA57B78B54704E256024E",
                "name": "Pancake Router",
                "tags": ["dex_router", "router", "benign_baseline"]
            }
        ],
        "dex_factories": [
            {
                "address": "0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73",
                "name": "Pancake Factory",
                "tags": ["dex_factory", "factory", "benign_baseline"]
            },
            {
                "address": "0x1CE0c2827e2eF14D5C4f29a091d735A204794041",
                "name": "Pancake LP Helper/Factory Related",
                "tags": ["dex_factory", "helper", "benign_baseline"]
            }
        ],
        "bridges": [],
        "lending_core": [],
        "oracles": [],
        "vaults": [],
    },

    "polygon": {
        "system": [
            {
                "address": "0x0000000000000000000000000000000000000000",
                "name": "Zero Address",
                "tags": ["system", "null", "benign_baseline"]
            },
            {
                "address": "0x000000000000000000000000000000000000dEaD",
                "name": "Dead Address",
                "tags": ["system", "burn", "benign_baseline"]
            },
        ],
        "wrapped_native": [
            {
                "address": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",
                "name": "WMATIC",
                "tags": ["wrapped_native", "token", "benign_baseline"]
            }
        ],
        "stablecoins": [
            {
                "address": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
                "name": "USDC",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0xc2132D05D31c914a87C6611C10748AEb04B58e8F",
                "name": "USDT",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0xdAb529f40E671A1D4bF91361c21bf9f0C9712ab7",
                "name": "BUSD",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063",
                "name": "DAI",
                "tags": ["stablecoin", "token", "benign_baseline"]
            }
        ],
        "bluechip_tokens": [
            {
                "address": "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619",
                "name": "WETH (Polygon)",
                "tags": ["bluechip", "token", "benign_baseline"]
            },
            {
                "address": "0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6",
                "name": "WBTC (Polygon)",
                "tags": ["bluechip", "token", "benign_baseline"]
            }
        ],
        "dex_routers": [
            {
                "address": "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff",
                "name": "QuickSwap Router",
                "tags": ["dex_router", "router", "benign_baseline"]
            }
        ],
        "dex_factories": [],
        "bridges": [],
        "lending_core": [],
        "oracles": [],
        "vaults": [],
    },

    "avax": {
        "system": [
            {
                "address": "0x0000000000000000000000000000000000000000",
                "name": "Zero Address",
                "tags": ["system", "null", "benign_baseline"]
            },
            {
                "address": "0x000000000000000000000000000000000000dEaD",
                "name": "Dead Address",
                "tags": ["system", "burn", "benign_baseline"]
            },
        ],
        "wrapped_native": [
            {
                "address": "0xB31f66AA3C1e785363F0875A1B74E27b85FD66c7",
                "name": "WAVAX",
                "tags": ["wrapped_native", "token", "benign_baseline"]
            }
        ],
        "stablecoins": [
            {
                "address": "0xB97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E",
                "name": "USDC",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0x9702230A8Ea53601f5cD2dc00fDBc13d4dF4A8c7",
                "name": "USDT",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0xA7D7079b0FEaD91F3e65f86E8915Cb59c1a4C664",
                "name": "USDC.e / Bridged Stable",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0xc7198437980c041c805A1EDcbA50c1Ce5db95118",
                "name": "USDT.e / Bridged Stable",
                "tags": ["stablecoin", "token", "benign_baseline"]
            }
        ],
        "bluechip_tokens": [
            {
                "address": "0x19860CCB0A68fd4213aB9D8266F7bBf05A8dDe98",
                "name": "WBTC.e",
                "tags": ["bluechip", "token", "benign_baseline"]
            },
            {
                "address": "0x2b2C81e08f1Af8835a78Bb2A90AE924ACE0eA4bE",
                "name": "sAVAX / Native Staked Derivative",
                "tags": ["bluechip", "token", "known_component"]
            }
        ],
        "dex_routers": [
            {
                "address": "0x60aE616a2155Ee3d9A68541Ba4544862310933d4",
                "name": "Trader Joe Router",
                "tags": ["dex_router", "router", "benign_baseline"]
            }
        ],
        "dex_factories": [],
        "bridges": [],
        "lending_core": [],
        "oracles": [],
        "vaults": [],
    },

    "opt": {
        "system": [
            {
                "address": "0x0000000000000000000000000000000000000000",
                "name": "Zero Address",
                "tags": ["system", "null", "benign_baseline"]
            },
            {
                "address": "0x000000000000000000000000000000000000dEaD",
                "name": "Dead Address",
                "tags": ["system", "burn", "benign_baseline"]
            },
            {
                "address": "0x4200000000000000000000000000000000000006",
                "name": "WETH (OP System)",
                "tags": ["wrapped_native", "token", "benign_baseline", "system_like"]
            }
        ],
        "wrapped_native": [],
        "stablecoins": [
            {
                "address": "0x7F5c764cBc14f9669B88837ca1490cCa17c31607",
                "name": "USDC",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0x94b008aA00579c1307B0EF2c499aD98a8ce58e58",
                "name": "USDT",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0xC22885e06cd8507c5c74a948C59af853AEd1Ea5C",
                "name": "Stable Asset",
                "tags": ["stablecoin", "token", "benign_baseline"]
            }
        ],
        "bluechip_tokens": [
            {
                "address": "0x68f180fcCe6836688e9084f035309E29Bf0A2095",
                "name": "WBTC",
                "tags": ["bluechip", "token", "benign_baseline"]
            }
        ],
        "dex_routers": [
            {
                "address": "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",
                "name": "Router",
                "tags": ["dex_router", "router", "benign_baseline"]
            }
        ],
        "dex_factories": [
            {
                "address": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
                "name": "Factory-like",
                "tags": ["dex_factory", "factory", "benign_baseline"]
            }
        ],
        "bridges": [],
        "lending_core": [],
        "oracles": [],
        "vaults": [],
    },

    "fantom": {
        "system": [
            {
                "address": "0x0000000000000000000000000000000000000000",
                "name": "Zero Address",
                "tags": ["system", "null", "benign_baseline"]
            },
            {
                "address": "0x000000000000000000000000000000000000dEaD",
                "name": "Dead Address",
                "tags": ["system", "burn", "benign_baseline"]
            },
        ],
        "wrapped_native": [],
        "stablecoins": [
            {
                "address": "0x04068DA6C83AFCFA0e13ba15A6696662335D5B75",
                "name": "USDC",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0x8D11eC38a3EB5E956B052f67Da8Bdc9bef8Abf3E",
                "name": "DAI",
                "tags": ["stablecoin", "token", "benign_baseline"]
            }
        ],
        "bluechip_tokens": [
            {
                "address": "0x321162Cd933E2Be498Cd2267a90534A804051b11",
                "name": "WBTC",
                "tags": ["bluechip", "token", "benign_baseline"]
            },
            {
                "address": "0x511D35c52a3C244E7b8bd92c0C297755FbD89212",
                "name": "WFTM / Native Related",
                "tags": ["wrapped_native", "token", "benign_baseline"]
            }
        ],
        "dex_routers": [
            {
                "address": "0xF491e7B69E4244ad4002BC14e878a34207E38c29",
                "name": "SpookySwap Router",
                "tags": ["dex_router", "router", "benign_baseline"]
            },
            {
                "address": "0xe1146b9AC456fCbB60644c36Fd3F868A9072fc6E",
                "name": "Router / DEX Component",
                "tags": ["dex_router", "router", "benign_baseline"]
            }
        ],
        "dex_factories": [],
        "bridges": [],
        "lending_core": [],
        "oracles": [],
        "vaults": [],
    },

    "arbi": {
        "system": [
            {
                "address": "0x0000000000000000000000000000000000000000",
                "name": "Zero Address",
                "tags": ["system", "null", "benign_baseline"]
            },
            {
                "address": "0x000000000000000000000000000000000000dEaD",
                "name": "Dead Address",
                "tags": ["system", "burn", "benign_baseline"]
            },
            {
                "address": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
                "name": "WETH",
                "tags": ["wrapped_native", "token", "benign_baseline"]
            }
        ],
        "wrapped_native": [],
        "stablecoins": [
            {
                "address": "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
                "name": "USDC",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0xd4d42F0b6DEF4CE0383636770eF773390d85c61A",
                "name": "USDT",
                "tags": ["stablecoin", "token", "benign_baseline"]
            },
            {
                "address": "0xDA10009cBd5D07dd0CeCc66161FC93D7c9000da1",
                "name": "DAI",
                "tags": ["stablecoin", "token", "benign_baseline"]
            }
        ],
        "bluechip_tokens": [],
        "dex_routers": [
            {
                "address": "0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506",
                "name": "Router",
                "tags": ["dex_router", "router", "benign_baseline"]
            }
        ],
        "dex_factories": [
            {
                "address": "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f",
                "name": "Factory-like",
                "tags": ["dex_factory", "factory", "benign_baseline"]
            }
        ],
        "bridges": [],
        "lending_core": [],
        "oracles": [],
        "vaults": [],
    },

    "base": {
        "system": [
            {
                "address": "0x0000000000000000000000000000000000000000",
                "name": "Zero Address",
                "tags": ["system", "null", "benign_baseline"]
            },
            {
                "address": "0x000000000000000000000000000000000000dEaD",
                "name": "Dead Address",
                "tags": ["system", "burn", "benign_baseline"]
            },
            {
                "address": "0x4200000000000000000000000000000000000006",
                "name": "WETH (Base System)",
                "tags": ["wrapped_native", "token", "benign_baseline", "system_like"]
            }
        ],
        "wrapped_native": [],
        "stablecoins": [],
        "bluechip_tokens": [],
        "dex_routers": [
            {
                "address": "0xfCD3842f85ed87ba2889b4D35893403796e67FF1",
                "name": "Router",
                "tags": ["dex_router", "router", "benign_baseline"]
            }
        ],
        "dex_factories": [],
        "bridges": [],
        "lending_core": [],
        "oracles": [],
        "vaults": [],
    },
}


# =====================================================================
#                    Registry Helpers
# =====================================================================

def get_common_address_registry(chain):
    chain = normalize_chain_name(chain)
    return COMMON_ADDRESS_REGISTRY.get(chain, {})


def get_common_address_meta(chain, address):
    chain = normalize_chain_name(chain)
    address = (address or "").lower()
    registry = COMMON_ADDRESS_REGISTRY.get(chain, {})

    for category, items in registry.items():
        for item in items:
            if item["address"].lower() == address:
                return {
                    "category": category,
                    **item
                }
    return None


def get_common_address(chain):
    """
    向后兼容：返回当前链的平面地址列表（原始大小写保留）。
    默认返回 registry 中的所有地址，不区分 benign / known_component。
    """
    chain = normalize_chain_name(chain)
    registry = COMMON_ADDRESS_REGISTRY.get(chain, {})

    result = []
    seen = set()

    for _, items in registry.items():
        for item in items:
            addr = item["address"]
            if addr.lower() not in seen:
                result.append(addr)
                seen.add(addr.lower())

    return result


def get_common_address_lower(chain, include_tags=None, exclude_tags=None, only_benign=False):
    """
    推荐新接口：
    - only_benign=True: 只返回适合用于“降噪 / 常见基础设施判断”的地址
    - include_tags / exclude_tags: 用于按语义筛选子集
    """
    chain = normalize_chain_name(chain)
    registry = COMMON_ADDRESS_REGISTRY.get(chain, {})

    include_tags = set(include_tags or [])
    exclude_tags = set(exclude_tags or [])

    result = []
    seen = set()

    for _, items in registry.items():
        for item in items:
            addr = item["address"].lower()
            tags = set(item.get("tags", []))

            if only_benign and "benign_baseline" not in tags:
                continue
            if include_tags and not (tags & include_tags):
                continue
            if exclude_tags and (tags & exclude_tags):
                continue

            if addr not in seen:
                result.append(addr)
                seen.add(addr)

    return result


def get_factory_address(chain):
    """
    向后兼容：优先从 registry 中返回第一个 dex_factory；
    如果没有则返回 None。
    """
    chain = normalize_chain_name(chain)
    registry = COMMON_ADDRESS_REGISTRY.get(chain, {})
    items = registry.get("dex_factories", [])
    if not items:
        return None
    return items[0]["address"]


# =====================================================================
#                    Other Existing Helpers
# =====================================================================

def get_rpc_endpoints(chain):
    chain = normalize_chain_name(chain)
    rpc_endpoints = {
        'eth': 'https://rpc.ankr.com/eth',
        'bsc': 'https://rpc.ankr.com/bsc',
        'polygon': 'https://polygon-rpc.com',
        'avax': 'https://api.avax.network/ext/bc/C/rpc',
        'fantom': 'https://rpc.fantom.network',
        'arbi': 'https://arbitrum.meowrpc.com',
        'pulsechain': 'https://rpc.pulsechain.com',
        'base': 'https://mainnet.base.org',
        'opt': 'https://mainnet.optimism.io',
        'gnosis': 'https://rpc.gnosischain.com/',
    }
    return rpc_endpoints.get(chain)


def get_function_calls_to_expand():
    return {
        'flashloan': [
            'flashLoan',
            'flashLoanSimple',
            'flashBorrow',
            'dYdXFlashLoan',
            'balancerFlashLoan',
            'flashMint',
            'maxFlashLoan',
            'flashFee',
            'flashRedeem',
            'depositFlashloan',
            'withdrawFlashloan'
        ],
        'callback': [
            'executeOperation',
            'onFlashLoan',
            'DVMFlashLoanCall',
            'DPPFlashLoanCall',
            'onERC3156FlashLoan',
            'executeFlashLoan',
            'onFlashLoanComplete',
            'onFlashLoanStart',
            'uniswapV2Call',
            'swap',
            'execute',
            'callFunction',
            'flashCallback',
            'receiveFlashLoan',
            'onDeferredLiquidityCheck',
            'flashClose',
            'onMorphoFlashLoan'
        ]
    }


def is_function_hash(function_name):
    if function_name.startswith('0x'):
        function_name = function_name[2:]
    function_hash_pattern = r'[0-9a-f]{8}'
    if re.match(function_hash_pattern, function_name):
        return True
    return False


def build_agent(task, model):
    from evotx.utils.agent import OllmaAgent, GPTAgent

    model = model.strip().lower()

    if model.startswith('gpt'):
        return GPTAgent(task, model)
    else:
        return OllmaAgent(task, model)
