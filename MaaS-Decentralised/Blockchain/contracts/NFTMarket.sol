// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "./MaaSCore.sol"; // Import MaaSCore contract

contract Market {

    IERC20 public paymentToken;  // ERC20 token contract
    address public admin;

    struct SellRequest {
        uint256 tokenId;
        uint256 price;
        address seller;
        bool isSold;
    }

    SellRequest[] public market;  // Dynamic array for the market

    // Events
    event NFTListedForSale(uint256 indexed tokenId, uint256 price, address indexed seller);
    event NFTPurchased(uint256 indexed tokenId, address indexed buyer);

    // Constructor
    constructor(IERC20 _paymentToken) {
        admin = msg.sender;
        paymentToken = _paymentToken != IERC20(address(0)) ? _paymentToken : IERC20(address(0));  // Use zero address if no token is provided
    }

    // Modifiers
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can perform this action");
        _;
    }

    modifier onlyNFTOwner(ERC721 nftContract, uint256 tokenId) {
        require(nftContract.ownerOf(tokenId) == msg.sender, "Only the owner of the NFT can perform this action");
        _;
    }

    // Function to list an NFT for sale on the market
    function listNFTForSale(MaaSCore nftContract, uint256 tokenId, uint256 price) public onlyNFTOwner(nftContract, tokenId) {
        require(price > 0, "Price must be greater than zero");

        market.push(SellRequest({
            tokenId: tokenId,
            price: price,
            seller: msg.sender,
            isSold: false
        }));

        emit NFTListedForSale(tokenId, price, msg.sender);
    }

    // Function to purchase an NFT from the market
    function purchaseNFT(MaaSCore nftContract, uint256 marketIndex) public {
        require(marketIndex < market.length, "Invalid market index");
        SellRequest storage sellRequest = market[marketIndex];

        require(!sellRequest.isSold, "This NFT is already sold");

        // Transfer the payment
        require(
            paymentToken.transferFrom(msg.sender, sellRequest.seller, sellRequest.price),
            "Payment failed"
        );

        // Transfer the NFT ownership
        nftContract.safeTransferFrom(sellRequest.seller, msg.sender, sellRequest.tokenId);
        sellRequest.isSold = true;

        emit NFTPurchased(sellRequest.tokenId, msg.sender);
    }

    // Function to search for offers in the market that match the request details
  /*  function searchMarket(
        MaaSCore nftContract,
        uint32 origin_x_coordinate,
        uint32 origin_y_coordinate,
        uint32 destination_x_coordinate,
        uint32 destination_y_coordinate,
        uint32 start_time,
        uint8 mode
    ) public view returns (uint256[] memory) {
        uint256[] memory matchingTokens = new uint256[](market.length);
        uint256 count = 0;

        for (uint256 i = 0; i < market.length; i++) {
            if (!market[i].isSold) {
                uint256 tokenId = market[i].tokenId;
                MaaSCore.Request memory request = nftContract.requests(tokenId);

                // Check if the request details match
                if (
                    request.origin_x_coordinate == origin_x_coordinate &&
                    request.origin_y_coordinate == origin_y_coordinate &&
                    request.destination_x_coordinate == destination_x_coordinate &&
                    request.destination_y_coordinate == destination_y_coordinate &&
                    request.start_time == start_time &&
                    request.mode == mode
                ) {
                    matchingTokens[count] = tokenId;
                    count++;
                }
            }
        }

        // Resize the array to the actual number of matches
        uint256[] memory result = new uint256[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = matchingTokens[i];
        }

        return result;
    }
*/
}
