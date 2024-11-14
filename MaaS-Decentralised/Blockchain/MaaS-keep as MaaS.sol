// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";

contract MaaS is ERC721URIStorage {

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
    
    
    mapping(uint256 => mapping(uint256 => Offer)) public carShareOffers;
    mapping(uint256 => mapping(uint256 => Offer)) public bikeShareOffers;
    mapping(uint256 => mapping(uint256 => Offer)) public publicTransportOffers;

    mapping(uint256 => Auction) public auctions;
  
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

    // Functions for NFT
    function mintRequestNFT(uint256 requestId, string memory tokenURI) internal returns (uint256) {
        _tokenIdCounter += 1;
        uint256 newItemId = _tokenIdCounter;
        _mint(msg.sender, newItemId);
        _setTokenURI(newItemId, tokenURI);
        return newItemId;
    }
function submitOffer(Offer memory newOffer) public {
        Provider memory provider = providers[msg.sender];
        Request memory request = requests[newOffer.request_id];

        require(isActiveAuction(newOffer.auction_id), "Auction is not active");

        // Check if provider's mode type matches the request's mode requirements
        if (request.mode == uint8(typeMode.CarShare)) {
            require(provider.mode_type == typeMode.CarShare, "Only CarShare providers can submit offers");
            carShareOffers[newOffer.auction_id][newOffer.provider_id] = newOffer;
        } else if (request.mode == uint8(typeMode.BikeShare)) {
            require(provider.mode_type == typeMode.BikeShare, "Only BikeShare providers can submit offers");
            bikeShareOffers[newOffer.auction_id][newOffer.provider_id] = newOffer;
        } else if (request.mode == uint8(typeMode.PublicTransport)) {
            require(provider.mode_type == typeMode.PublicTransport, "Only PublicTransport providers can submit offers");
            publicTransportOffers[newOffer.auction_id][newOffer.provider_id] = newOffer;
        }

        emit OfferSubmitted(uint32(newOffer.request_id), uint32(newOffer.provider_id), newOffer.price);
    }

    // Function to accept offer and make payment
    function acceptOffer(uint256 offerId, uint256 mode,uint256 request_id, uint256 provider_id, string memory tokenURI) public {
        // Assuming the request is being accepted by the commuter
        Request storage request = requests[request_id];
        require(request.status == statuses.active, "Request is not active");

        address provider_address = providerIdToAddress[provider_id];

        // Fetch the accepted offer details
        Offer memory acceptedOffer;
        if (mode == 1) {
            acceptedOffer = carShareOffers[request.winningOfferId][provider_id];
        } else if (mode == 2) {
            acceptedOffer = bikeShareOffers[request.winningOfferId][provider_id];
        } else if (mode == 3) {
            acceptedOffer = publicTransportOffers[request.winningOfferId][provider_id];
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
// Function to search for offers in the market that match the request details
    
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
function finalizeAuction(uint256 auction_id) public  {
        require(isActiveAuction(auction_id), "Auction is not active or has not started yet");
        Auction storage auction = auctions[auction_id];
        require(block.timestamp >= auction.auction_start_time + 3 hours, "Auction cannot be finalized before 3 hours");

        Request memory auctionRequest = requests[auction.request_id];
        uint8 mode = auctionRequest.mode;

        // Array to hold relevant offers
        Offer[] memory relevantOffers = new Offer[](3);
        uint256 offerCount = 0;

        // Collect offers based on the mode
        if (mode == uint8(typeMode.CarShare)) {
            relevantOffers[offerCount] = carShareOffers[auction_id][auction.provider_id];
            offerCount++;
        }

        if (mode == uint8(typeMode.BikeShare) ) {
            relevantOffers[offerCount] = bikeShareOffers[auction_id][auction.provider_id];
            offerCount++;
        }

        if (mode == uint8(typeMode.PublicTransport)) {
            relevantOffers[offerCount] = publicTransportOffers[auction_id][auction.provider_id];
            offerCount++;
        }

        require(offerCount > 0, "No offers available to finalize");
        auction.status = true;

        // Further logic for selecting the winner based on the offers
    }/*
    function finalizeAuction(uint256 auction_id) public onlyAdmin {
        require(isActiveAuction(auction_id), "Auction is not active or has not started yet");
        Auction storage auction = auctions[auction_id];
        require(block.timestamp >= auction.auction_start_time + 3 hours, "Auction cannot be finalized before 3 hours");

        Request memory auctionRequest = requests[auction.request_id];
        uint8 mode = auctionRequest.mode;

        Offer[] memory relevantOffers;
        uint256 offerCount = 1;

        // Select the relevant offers based on the mode
        if (mode == 1 ) {
            if (carShareOffers[i].auction_id == auction_id) {
                relevantOffers[0]= carShareOffers[auction_id].;
                
            } 
        }
        for (uint256 i = 0; i < bikeShareOffers.length; i++) {
            if (bikeShareOffers[i].auction_id == auction_id) {
                relevantOffers[1]= bikeShareOffers[i];
                
            }
      
         } else if (mode == 2 || mode == 4|| mode == 6|| mode == 7) {
            for (uint256 i = 0; i < bikeShareOffers.length; i++) {
            if (bikeShareOffers[i].auction_id == auction_id) {
                relevantOffers[0]= bikeShareOffers[i];
                
            }
        }
         } else if (mode == 3) {
            for (uint256 i = 0; i < publicTransportOffers.length; i++) {
            if (publicTransportOffers[i].auction_id == auction_id) {
                relevantOffers[0]= publicTransportOffers[i];
                
            }
        }
        
         }   for (uint256 i = 0; i < bikeShareOffers.length; i++) {
            if (bikeShareOffers[i].auction_id == auction_id) {
                relevantOffers[1]= bikeShareOffers[i];
                
            }
            }

 
            offerCount = 2;
        } else if (mode == 5) {
            for (uint256 i = 0; i < carShareOffers.length; i++) {
            if (carShareOffers[i].auction_id == auction_id) {
                relevantOffers[0]= carShareOffers[i];
                
            }
            }

            for (uint256 i = 0; i < publicTransportOffers.length; i++) {
            if (publicTransportOffers[i].auction_id == auction_id) {
                relevantOffers[1]= publicTransportOffers[i];
                
            }
            }

             offerCount = 2;
        } else if (mode == 6) {
            for (uint256 i = 0; i < bikeShareOffers.length; i++) {
            if (bikeShareOffers[i].auction_id == auction_id) {
                relevantOffers[0]= bikeShareOffers[i];
                
            }
            }

            for (uint256 i = 0; i < publicTransportOffers.length; i++) {
            if (publicTransportOffers[i].auction_id == auction_id) {
                relevantOffers[1]= publicTransportOffers[i];
                
            }
            }
 
            offerCount = 2;
        } else if (mode == 7) {
            for (uint256 i = 0; i < bikeShareOffers.length; i++) {
            if (bikeShareOffers[i].auction_id == auction_id) {
                relevantOffers[0]=  bikeShareOffers[i] ;
                
            }
            }

            for (uint256 i = 0; i < publicTransportOffers.length; i++) {
            if (publicTransportOffers[i].auction_id == auction_id) {
                relevantOffers[1]= publicTransportOffers[i];
                
            }
            }
                       for (uint256 i = 0; i < carShareOffers.length; i++) {
            if (carShareOffers[i].auction_id == auction_id) {
                relevantOffers[2]= carShareOffers[i];
                
            }
            }
 
  
            offerCount = 3;
        }

        require(offerCount > 0, "No offers available to finalize");
        auction.status = true;
    }
*/
     

}
