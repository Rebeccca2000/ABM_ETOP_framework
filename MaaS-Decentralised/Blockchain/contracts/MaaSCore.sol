// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract MaaSCore is ERC721URIStorage {

    IERC20 public paymentToken;  // ERC20 token contract

    enum income {
        low,
        middle,
        high
    }

    enum health {
        good,
        poor
    }

    enum typeMode {
        CarShare,
        BikeShare,
        PublicTransport
    }

    enum purpose {
        work,
        school,
        shopping,
        medical
    }

    enum statuses {
        active,
        closed,
        cancelled,
        serviceSelected
    }

    struct Commuter {
        uint256 commuter_id;
        uint256 x_coordinate;
        uint256 y_coordinate;
        income c_income;
        typeMode preferred_mode_id;
        uint32 age;
        bool has_disablility;
        bool tech_access;
        health c_health;
    }

    struct Provider {
        uint256 provider_id;
        address p_address;
        typeMode mode_type;
    }

    struct Request {
        uint256 request_id;
        uint256 commuter_id;
        uint32 origin_x_coordinate;
        uint32 origin_y_coordinate;
        uint32 destination_x_coordinate;
        uint32 destination_y_coordinate;
        uint32 start_time;
        purpose travelPurpose;
        statuses status;
        uint32 winningOfferId;
        uint8 mode;
    }

    struct Offer {
        uint256 id;
        uint256 request_id;
        uint256 provider_id;
        uint256 auction_id;
        uint256 price;
        typeMode mode;
        uint256 startTime;
        uint256 totalTime;
        uint256 totalPrice;
        uint256 totalLength;
    }

    struct Auction {
        uint256 auction_id;
        uint32 modeIndicator;
        string path;
        bool status;
        uint256 request_id;
        uint256 commuter_id;
        uint256[] winner;
        uint256 auction_start_time;
        uint256 provider_id;
    }

    struct SellRequest {
        uint256 tokenId;
        uint256 price;
        address seller;
        bool isSold;
    }

    // Events
    event CommuterAdded(
        uint32 commuter_id,
        uint32 x_coordinate,
        uint32 y_coordinate,
        income c_income,
        uint32 age,
        bool has_disablility,
        bool tech_access,
        health c_health,
        typeMode operational_mode,
        address indexed c_address
    );

    event ProviderAdded(
        uint32 provider_id,
        address p_address,
        typeMode mode_type,
        string company_name
    );

    event RequestAdded(
        uint32 request_id,
        uint32 commuter_id,
        uint32 origin_x_coordinate,
        uint32 origin_y_coordinate,
        uint32 destination_x_coordinate,
        uint32 destination_y_coordinate,
        uint32 start_time,
        purpose travelPurpose,
        statuses status,
        bool fulfilled,
        uint8 mode
    );

    event OfferSubmitted(uint32 request_id, uint32 provider_id, uint256 price);
    event OfferAccepted(uint32 request_id, uint32 provider_id, uint256 price);
    event OfferDeclined(uint32 request_id, uint32 provider_id);
    event RideSold(uint32 request_id, uint256 price);
    event RequestBought(uint32 request_id, uint256 price);
    event NFTListedForSale(uint256 indexed tokenId, uint256 price, address indexed seller);
    event NFTPurchased(uint256 indexed tokenId, address indexed buyer);

    // State variables
    mapping(address => Commuter) public commuters;
    mapping(address => Provider) public providers;
    mapping(uint256 => Request) public requests;

    mapping(uint256 => address) public providerIdToAddress;
    
    Offer[] public carShareOffers;
    Offer[] public bikeShareOffers;
    Offer[] public publicTransportOffers;

    mapping(uint256 => Auction) public auctions;
    SellRequest[] public market;  // Dynamic array for the market

    uint256 public _numberOfAuctions = 0;
    uint256 public _numberOfCommuters = 0;
    uint256 public _numberproviders = 0;
    uint256 public _numberOfRequests = 0;
    uint256 private _tokenIdCounter = 0;  // Manual counter for token IDs
    address public admin;

    // Constructor
   constructor(IERC20 _paymentToken) ERC721("MaaS Request NFT", "MRNFT") {
    admin = msg.sender;
    paymentToken = _paymentToken != IERC20(address(0)) ? _paymentToken : IERC20(address(0));  // Use zero address if no token is provided
}
    // Modifiers
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can perform this action");
        _;
    }

    modifier onlyNFTOwner(uint256 tokenId) {
        require(ownerOf(tokenId) == msg.sender, "Only the owner of the NFT can perform this action");
        _;
    }

    // Functions for NFT
    function mintRequestNFT(uint256 requestId, string memory tokenURI) internal returns (uint256) {
        _tokenIdCounter += 1;
        uint256 newItemId = _tokenIdCounter;
        _mint(msg.sender, newItemId);
        _setTokenURI(newItemId, tokenURI);
        return newItemId;
    }

    // Function to accept offer and make payment
    function acceptOffer(uint256 request_id, uint256 provider_id, string memory tokenURI) public {
        // Assuming the request is being accepted by the commuter
        Request storage request = requests[request_id];
        require(request.status == statuses.active, "Request is not active");

        address provider_address = providerIdToAddress[provider_id];

        // Fetch the accepted offer details
        Offer memory acceptedOffer;
        if (request.mode == 1) {
            acceptedOffer = carShareOffers[request.winningOfferId];
        } else if (request.mode == 2) {
            acceptedOffer = bikeShareOffers[request.winningOfferId];
        } else if (request.mode == 3) {
            acceptedOffer = publicTransportOffers[request.winningOfferId];
        }

        // Transfer payment to the provider
        require(
            paymentToken.transferFrom(msg.sender, provider_address, acceptedOffer.price),
            "Payment failed"
        );

        // Mint the NFT for the accepted request
        uint256 nftId = mintRequestNFT(request_id, tokenURI);

        // Update request status and emit events
        request.status = statuses.serviceSelected;
        emit OfferAccepted(uint32(request_id), uint32(provider_id), acceptedOffer.price);
    }

    // New function to list an NFT for sale on the market
    function listNFTForSale(uint256 tokenId, uint256 price) public onlyNFTOwner(tokenId) {
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
    function purchaseNFT(uint256 marketIndex) public {
        require(marketIndex < market.length, "Invalid market index");
        SellRequest storage sellRequest = market[marketIndex];

        require(!sellRequest.isSold, "This NFT is already sold");

        // Transfer the payment
        require(
            paymentToken.transferFrom(msg.sender, sellRequest.seller, sellRequest.price),
            "Payment failed"
        );

        // Transfer the NFT ownership
        _transfer(sellRequest.seller, msg.sender, sellRequest.tokenId);
        sellRequest.isSold = true;

        emit NFTPurchased(sellRequest.tokenId, msg.sender);
    }

    // Function to search for offers in the market that match the request details
    function searchMarket(uint32 origin_x_coordinate, uint32 origin_y_coordinate, uint32 destination_x_coordinate, uint32 destination_y_coordinate, uint32 start_time, uint8 mode) public view returns (uint256[] memory) {
        uint256[] memory matchingTokens = new uint256[](market.length);
        uint256 count = 0;

        for (uint256 i = 0; i < market.length; i++) {
            if (!market[i].isSold) {
                uint256 tokenId = market[i].tokenId;
                Request memory request = requests[tokenId];

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

    // Other MaaS functions (e.g., addCommuter, addProvider, addRequest, submitOffer, etc.)

    function isActiveAuction(uint256 auction_id) public view returns (bool) {
        require(auction_id <= _numberOfAuctions && auctions[auction_id].auction_start_time + 1 hours >= block.timestamp, "Invalid Auction Id");
        return true;
    }

    function addCommuter(Commuter memory newCommuter) public {
        commuters[msg.sender] = newCommuter;
        _numberOfCommuters = _numberOfCommuters + 1;
        emit CommuterAdded(
            uint32(newCommuter.commuter_id),
            uint32(newCommuter.x_coordinate),
            uint32(newCommuter.y_coordinate),
            newCommuter.c_income,
            newCommuter.age,
            newCommuter.has_disablility,
            newCommuter.tech_access,
            newCommuter.c_health,
            newCommuter.preferred_mode_id,
            msg.sender
        );
    }

    function addProvider(uint256 provider_id, address provider_address, Provider memory newProvider) public {
        providers[provider_address] = newProvider;
        providerIdToAddress[provider_id] = provider_address;
        _numberproviders = _numberproviders + 1;
        emit ProviderAdded(
            uint32(provider_id),
            provider_address,
            newProvider.mode_type,
            "" // Assuming company_name is not provided; replace with actual value if available
        );
    }

    function addRequest(Request memory newRequest) public {
        require(
            newRequest.mode >= 0 && newRequest.mode <= 7,
            "Invalid mode value"
        );
        requests[newRequest.request_id] = newRequest;
        _numberOfRequests = _numberOfRequests + 1;

        emit RequestAdded(
            uint32(newRequest.request_id),
            uint32(newRequest.commuter_id),
            newRequest.origin_x_coordinate,
            newRequest.origin_y_coordinate,
            newRequest.destination_x_coordinate,
            newRequest.destination_y_coordinate,
            newRequest.start_time,
            newRequest.travelPurpose,
            newRequest.status,
            false, // Assuming the request is not fulfilled upon creation
            newRequest.mode
        );
    }

    function submitOffer(Offer memory newOffer) public {
        Provider memory provider = providers[msg.sender];
        Request memory request = requests[newOffer.request_id];

        require(isActiveAuction(newOffer.auction_id), "Auction is not active");

        // Check if provider's mode type matches the request's mode requirements
        if (request.mode == 1) {
            require(provider.mode_type == typeMode.CarShare, "Only CarShare providers can submit offers");
        } else if (request.mode == 2) {
            require(provider.mode_type == typeMode.BikeShare, "Only BikeShare providers can submit offers");
        } else if (request.mode == 3) {
            require(provider.mode_type == typeMode.PublicTransport, "Only PublicTransport providers can submit offers");
        } else if (request.mode == 4) {
            require(provider.mode_type == typeMode.CarShare || provider.mode_type == typeMode.BikeShare, "Only CarShare or BikeShare providers can submit offers");
        } else if (request.mode == 5) {
            require(provider.mode_type == typeMode.CarShare || provider.mode_type == typeMode.PublicTransport, "Only CarShare or PublicTransport providers can submit offers");
        } else if (request.mode == 6) {
            require(provider.mode_type == typeMode.BikeShare || provider.mode_type == typeMode.PublicTransport, "Only BikeShare or PublicTransport providers can submit offers");
        } else if (request.mode == 7) {
            require(
                provider.mode_type == typeMode.CarShare || 
                provider.mode_type == typeMode.BikeShare || 
                provider.mode_type == typeMode.PublicTransport, 
                "Only CarShare, BikeShare, or PublicTransport providers can submit offers"
            );
        }

        // Initialize a variable to track whether a lower offer exists
        bool isLowerOffer = true;
        
        // Set the offer ID to the current length of the array, ensuring it matches the index
        if (provider.mode_type == typeMode.CarShare) {
            newOffer.id = carShareOffers.length;
            for (uint256 i = 0; i < carShareOffers.length; i++) {
                if (carShareOffers[i].auction_id == newOffer.auction_id) {
                    if (newOffer.price >= carShareOffers[i].price) {
                        isLowerOffer = false;
                        break;
                    }
                }
            }
            if (isLowerOffer) {
                carShareOffers.push(newOffer);
            }
        } else if (provider.mode_type == typeMode.BikeShare) {
            newOffer.id = bikeShareOffers.length;
            for (uint256 i = 0; i < bikeShareOffers.length; i++) {
                if (bikeShareOffers[i].auction_id == newOffer.auction_id) {
                    if (newOffer.price >= bikeShareOffers[i].price) {
                        isLowerOffer = false;
                        break;
                    }
                }
            }
            if (isLowerOffer) {
                bikeShareOffers.push(newOffer);
            }
        } else if (provider.mode_type == typeMode.PublicTransport) {
            newOffer.id = publicTransportOffers.length;
            for (uint256 i = 0; i < publicTransportOffers.length; i++) {
                if (publicTransportOffers[i].auction_id == newOffer.auction_id) {
                    if (newOffer.price >= publicTransportOffers[i].price) {
                        isLowerOffer = false;
                        break;
                    }
                }
            }
            if (isLowerOffer) {
                publicTransportOffers.push(newOffer);
            }
        }

        // Emit an event only if the offer was accepted and stored
        if (isLowerOffer) {
            emit OfferSubmitted(uint32(newOffer.request_id), uint32(newOffer.provider_id), newOffer.price);
        } else {
            revert("New offer price must be lower than the current submitted offer for this auction");
        }
    }

    function createAuction(Auction memory newAuction) public {
        _numberOfAuctions = _numberOfAuctions + 1;
        newAuction.auction_id = _numberOfAuctions;
        auctions[newAuction.auction_id] = newAuction;
    }

    function declineOffer(uint32 request_id, uint32 provider_id) public {
        emit OfferDeclined(request_id, provider_id);
    }

    function getNumberRequests() public view returns (uint256) {
        return _numberOfRequests;
    }

    function finalizeAuction(uint256 auction_id) public onlyAdmin {
        require(isActiveAuction(auction_id), "Auction is not active or has not started yet");
        Auction storage auction = auctions[auction_id];
        require(block.timestamp >= auction.auction_start_time + 3 hours, "Auction cannot be finalized before 3 hours");

        Request memory auctionRequest = requests[auction.request_id];
        uint8 mode = auctionRequest.mode;

        Offer[] memory relevantOffers;
        uint256 offerCount = 0;

        // Select the relevant offers based on the mode
        if (mode == 1) {
            relevantOffers = filterOffersByModeAndAuction(carShareOffers, auction_id);
            offerCount = relevantOffers.length;
        } else if (mode == 2) {
            relevantOffers = filterOffersByModeAndAuction(bikeShareOffers, auction_id);
            offerCount = relevantOffers.length;
        } else if (mode == 3) {
            relevantOffers = filterOffersByModeAndAuction(publicTransportOffers, auction_id);
            offerCount = relevantOffers.length;
        } else if (mode == 4) {
            relevantOffers = combineOffers(
                filterOffersByModeAndAuction(carShareOffers, auction_id),
                filterOffersByModeAndAuction(bikeShareOffers, auction_id)
            );
            offerCount = relevantOffers.length;
        } else if (mode == 5) {
            relevantOffers = combineOffers(
                filterOffersByModeAndAuction(carShareOffers, auction_id),
                filterOffersByModeAndAuction(publicTransportOffers, auction_id)
            );
            offerCount = relevantOffers.length;
        } else if (mode == 6) {
            relevantOffers = combineOffers(
                filterOffersByModeAndAuction(bikeShareOffers, auction_id),
                filterOffersByModeAndAuction(publicTransportOffers, auction_id)
            );
            offerCount = relevantOffers.length;
        } else if (mode == 7) {
            relevantOffers = combineOffers(
                combineOffers(
                    filterOffersByModeAndAuction(carShareOffers, auction_id),
                    filterOffersByModeAndAuction(bikeShareOffers, auction_id)
                ),
                filterOffersByModeAndAuction(publicTransportOffers, auction_id)
            );
            offerCount = relevantOffers.length;
        }

        require(offerCount > 0, "No offers available to finalize");

        sortOffersByPrice(relevantOffers);

        if (mode == 1 || mode == 2 || mode == 3) {
            auction.winner = [relevantOffers[0].id];
        } else if (mode == 4 || mode == 5 || mode == 6) {
            require(offerCount >= 2, "Not enough offers to select two values");
            auction.winner =[relevantOffers[0].id, relevantOffers[1].id];
        } else if (mode == 7) {
            require(offerCount >= 3, "Not enough offers to select three values");
            auction.winner =[relevantOffers[0].id,relevantOffers[1].id,relevantOffers[2].id];
        }

        auction.status = true;
    }

    function filterOffersByModeAndAuction(Offer[] storage offers, uint256 auction_id) internal view returns (Offer[] memory) {
        uint256 count = 0;
        for (uint256 i = 0; i < offers.length; i++) {
            if (offers[i].auction_id == auction_id) {
                count++;
                break;
            }
        }
        Offer[] memory filteredOffers = new Offer[](count);
        uint256 index = 0;
        for (uint256 i = 0; i < offers.length; i++) {
            if (offers[i].auction_id == auction_id) {
                filteredOffers[index] = offers[i];
                index++;
            }
        }
        return filteredOffers;
    }

    function combineOffers(Offer[] memory offers1, Offer[] memory offers2) internal pure returns (Offer[] memory) {
        Offer[] memory combined = new Offer[](offers1.length + offers2.length);
        for (uint256 i = 0; i < offers1.length; i++) {
            combined[i] = offers1[i];
        }
        for (uint256 i = 0; i < offers2.length; i++) {
            combined[offers1.length + i] = offers2[i];
        }
        return combined;
    }

    function sortOffersByPrice(Offer[] memory offers) internal pure {
        for (uint256 i = 0; i < offers.length - 1; i++) {
            for (uint256 j = 0; j < offers.length - i - 1; j++) {
                if (offers[j].price > offers[j + 1].price) {
                    Offer memory temp = offers[j];
                    offers[j] = offers[j + 1];
                    offers[j + 1] = temp;
                }
            }
        }
    }

}