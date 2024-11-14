const hre = require("hardhat");

async function main() {
    // Define the payment token address (replace with the actual token address if available)
    const paymentTokenAddress = "0x0000000000000000000000000000000000000000"; // Use a real ERC20 token address if you have one

    // Deploy the MaaSCore contract
    const MaaSCoreFactory = await hre.ethers.getContractFactory("MaaSCore");
    const maasCore = await MaaSCoreFactory.deploy(paymentTokenAddress);
    //await maasCore.deployed();
    console.log("MaaSCore deployed to:", maasCore.target);

    // Deploy the Market contract
    const MarketFactory = await hre.ethers.getContractFactory("Market");
    const market = await MarketFactory.deploy(paymentTokenAddress);
    //await market.deployed();
    console.log("Market deployed to:", market.target);

    // Optionally, you can link the Market contract to the MaaS1 contract
    // by calling functions on the Market contract and passing the MaaS1 address
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
