const { ethers } = require('ethers');

// Connect to Ethereum network
const ethereumProvider = new ethers.JsonRpcProvider("http://127.0.0.1:8545");
console.log('Provider initialized:', ethereumProvider);

const userAddress = '0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266';
const userPrivateKey = '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80';

// Contract Addresses (replace these with the actual deployed addresses)
const maasCoreAddress = '0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9';        // Replace with actual address
const nftMarketAddress = '0x5FC8d32690cc91D4c39d9d3abcBD16989F875707';      // Replace with actual address

// ABIs for each contract
const maasCoreABI = [
    "function addProvider(uint256 provider_id, address provider_address, (uint256 provider_id, address p_address, uint8 mode_type) newProvider)",
    "function addCommuter((uint256 commuter_id, uint256 x_coordinate, uint256 y_coordinate, uint8 c_income, uint8 preferred_mode_id, uint32 age, bool has_disability, bool tech_access, uint8 c_health) newCommuter)",
    "function addRequest((uint256 request_id, uint256 commuter_id, uint32 origin_x_coordinate, uint32 origin_y_coordinate, uint32 destination_x_coordinate, uint32 destination_y_coordinate, uint32 start_time, uint8 travelPurpose, uint8 status, uint32 winningOfferId, uint8 mode) newRequest)",
    "function getRequest(uint256 index) view returns (uint256, uint256, uint32, uint32, uint32, uint32, uint32, uint8, uint8, uint32, uint8)",
    "function getNumberRequests() view returns (uint256)",
    "function getNumberCommuters() view returns (uint256)",
    "function getNumberProviders() view returns (uint256)",
    "function createAuction(Auction memory newAuction)"
];

const nftMarketABI = [
    "function listNFTForSale(uint256 tokenId, uint256 price)",
    "function purchaseNFT(uint256 marketIndex)",
    "function searchMarket(uint32 origin_x_coordinate, uint32 origin_y_coordinate, uint32 destination_x_coordinate, uint32 destination_y_coordinate, uint32 start_time, uint8 mode) view returns (uint256[])"
];

// Initialize wallet and contract instances
const wallet = new ethers.Wallet(userPrivateKey, ethereumProvider);
const maasCoreContract = new ethers.Contract(maasCoreAddress, maasCoreABI, wallet);
const nftMarketContract = new ethers.Contract(nftMarketAddress, nftMarketABI, wallet);

// Define the new commuter object
const newCommuter = {
    commuter_id: 1,
    x_coordinate: 10,
    y_coordinate: 200,
    c_income: 1,  // Assuming 1 represents 'middle' income
    preferred_mode_id: 1, // Assuming 1 represents 'BikeShare'
    age: 30,
    has_disability: false,
    tech_access: true,
    c_health: 1 // Assuming 1 represents 'good' health
};

// Define the new provider object
const newProvider = {
    provider_id: 1,
    p_address: userAddress,
    mode_type: 2 // Assuming 2 represents 'PublicTransport'
};

// Define the new request object
const newRequest = {
    request_id: 1,
    commuter_id: 1,
    origin_x_coordinate: 0,
    origin_y_coordinate: 0,
    destination_x_coordinate: 13,
    destination_y_coordinate: 3,
    start_time: 1234567890,
    travelPurpose: 0, // Assuming 0 represents 'work'
    status: 0, // Assuming 0 represents 'active'
    winningOfferId: 0,
    mode: 2 // Assuming 2 represents 'PublicTransport'
};

// Function to add a commuter
async function addCommuter() {
    try {
        const tx = await maasCoreContract.addCommuter(newCommuter, {
            gasLimit: 1000000
        });
        await tx.wait();
        console.log('Commuter added:', newCommuter);
    } catch (error) {
        console.error('Error adding commuter:', error);
    }
}

// Function to add a provider
async function addProvider() {
    try {
        const tx = await maasCoreContract.addProvider(newProvider.provider_id, newProvider.p_address, newProvider);
        await tx.wait();
        console.log('Provider added:', newProvider);
    } catch (error) {
        console.error('Error adding provider:', error);
    }
}

// Function to add a request

// Function to add a request
async function addRequest() {
    const tx = await maasCoreContract.addRequest(newRequest);
    await tx.wait();

    // Call dijkstraWithoutDiagonals function
    const shortestRoute = dijkstraWithoutDiagonals(
        [newRequest.origin_x_coordinate, newRequest.origin_y_coordinate],
        [newRequest.destination_x_coordinate, newRequest.destination_y_coordinate]
    );

    const publicRoute = findOptimalRoute(
        [newRequest.origin_x_coordinate, newRequest.origin_y_coordinate],
        [newRequest.destination_x_coordinate, newRequest.destination_y_coordinate]
    );

    const detailsIternary = buildDetailedItinerary(publicRoute,
        [newRequest.origin_x_coordinate, newRequest.origin_y_coordinate],
        [newRequest.destination_x_coordinate, newRequest.destination_y_coordinate]
    );

    const timeandprice = calculateTotalTimeAndPricePublic(detailsIternary,
       1, 3, 3, 1.5, 1.5);
    const [total, totaltime, totalprice] = calculateSingleModeTimeAndPrice(
        [newRequest.origin_x_coordinate, newRequest.origin_y_coordinate],
        [newRequest.destination_x_coordinate, newRequest.destination_y_coordinate],
        3, 5
    );

    console.log('totaltime :', totaltime);
    console.log('totalprice :', totalprice);

    console.log('timeandprice :', timeandprice);

    console.log('details :', detailsIternary);

    console.log('Request added:', newRequest);
    console.log('public added:', publicRoute);
    const gasLimit = 1000000; // Adjust the gas limit as needed

    console.log('Shortest route:', shortestRoute);

    const singlenewAuction = {
        auction_id: 0,
        modeIndicator: 0,
        path: shortestRoute.toString(),
        status: false,
        request_id: newRequest.request_id,
        commuter_id: newRequest.commuter_id,
        winner: 0
    };

    const atx = await maasCoreContract.createAuction(singlenewAuction);
    await atx.wait();
    console.log('atx :', atx);
}

// Function to list an NFT for sale
async function listNFTForSale(tokenId, price) {
    try {
        const tx = await nftMarketContract.listNFTForSale(tokenId, price, {
            gasLimit: 1000000
        });
        await tx.wait();
        console.log(`NFT with tokenId ${tokenId} listed for sale at price ${ethers.utils.formatEther(price)} ETH`);
    } catch (error) {
        console.error('Error listing NFT for sale:', error);
    }
}

// Function to purchase an NFT
async function purchaseNFT(marketIndex) {
    try {
        const tx = await nftMarketContract.purchaseNFT(marketIndex, {
            gasLimit: 1000000
        });
        await tx.wait();
        console.log(`NFT at market index ${marketIndex} purchased`);
    } catch (error) {
        console.error('Error purchasing NFT:', error);
    }
}

// Function to search the market
async function searchMarket(origin_x, origin_y, destination_x, destination_y, start_time, mode) {
    try {
        const matchingTokens = await nftMarketContract.searchMarket(origin_x, origin_y, destination_x, destination_y, start_time, mode);
        console.log('Matching tokens:', matchingTokens);
        return matchingTokens;
    } catch (error) {
        console.error('Error searching market:', error);
    }
}

// Function to get the number of commuters
async function getNumberCommuters() {
    try {
        const numberOfCommuters = await maasCoreContract.getNumberCommuters();
        console.log('Number of commuters:', numberOfCommuters.toString());
    } catch (error) {
        console.error('Error getting number of commuters:', error);
    }
}

// Function to get the number of providers
async function getNumberProviders() {
    try {
        const numberOfProviders = await maasCoreContract.getNumberProviders();
        console.log('Number of providers:', numberOfProviders.toString());
    } catch (error) {
        console.error('Error getting number of providers:', error);
    }
}

// Function to get the number of requests
async function getNumberRequests() {
    try {
        const numberOfRequests = await maasCoreContract.getNumberRequests();
        console.log('Number of requests:', numberOfRequests.toString());
    } catch (error) {
        console.error('Error getting number of requests:', error);
    }
}

// Execute the functions
async function main() {
    await addCommuter();
    await addProvider();
    await addRequest();
    await getNumberCommuters();
    await getNumberProviders();
    await getNumberRequests();

    // List an NFT for sale
    await listNFTForSale(1, ethers.utils.parseEther("0.2"));

    // Search the market for matching requests
    const matchingTokens = await searchMarket(0, 0, 13, 3, 1234567890, 2);

    // If there are matching tokens, purchase the first one
    if (matchingTokens.length > 0) {
        await purchaseNFT(matchingTokens[0]);
    }
}

main().catch((error) => {
    console.error('Error executing functions:', error);
});

// Helper function to calculate the Euclidean distance between two points
const calculateDistance = (point1, point2) => {
    const [x1, y1] = point1;
    const [x2, y2] = point2;
    const deltaX = x2 - x1;
    const deltaY = y2 - y1;
    return Math.sqrt(deltaX ** 2 + deltaY ** 2);
}

// Helper function to calculate the total length of a route
const calculateTotalRouteLength = (route) => {
    let totalLength = 0.0;
    for (let i = 0; i < route.length - 1; i++) {
        totalLength += calculateDistance(route[i], route[i + 1]);
    }
    return totalLength;
}

const dijkstraWithoutDiagonals = (start, end) => {
    const moves = [[0, 1], [1, 0], [0, -1], [-1, 0]]; // Possible moves (right, down, left, up)
    const rows = 14, cols = 14;
    const minCost = Array.from({ length: rows }, () => Array(cols).fill(Infinity)); // Initialize min cost matrix with infinity
    minCost[start[0]][start[1]] = 0; // Cost to start is 0
    const predecessor = Array.from({ length: rows }, () => Array(cols).fill(null)); // Initialize predecessors
    const queue = [[0, start]]; // Priority queue with start point

    while (queue.length > 0) {
        queue.sort((a, b) => a[0] - b[0]); // Sort queue to simulate a priority queue
        const [currentCost, [x, y]] = queue.shift(); // Get the point with the minimum cost

        if (x === end[0] && y === end[1]) break; // Exit if we reach the end point

        for (const [dx, dy] of moves) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols) { // Ensure the move is within the grid
                const nextCost = currentCost + 1;
                if (nextCost < minCost[nx][ny]) {
                    minCost[nx][ny] = nextCost;
                    predecessor[nx][ny] = [x, y];
                    queue.push([nextCost, [nx, ny]]); // Add the next point to the queue
                }
            }
        }
    }

    const path = [];
    let step = end;
    while (step.toString() !== start.toString()) {
        path.push(step);
        step = predecessor[step[0]][step[1]]; // Trace back the path using predecessors
    }
    path.push(start);
    path.reverse(); // Reverse the path to get the correct order
    console.log(`Path is ${JSON.stringify(path)}`);
    return path; ////send this back to blockchain
}

const calculateSingleModeTimeAndPrice = (origin, destination, unitPrice, unitSpeed) => {
    const shortestRoute = dijkstraWithoutDiagonals(origin, destination);
    const totalLength = calculateTotalRouteLength(shortestRoute); // Calculate the total length of the route
    const totalPrice = unitPrice * totalLength; // Calculate the total price
    const totalTime = totalLength / unitSpeed; // Calculate the total time
    console.log(`Test calculate_single_mode_time_and_price: ${totalTime}, ${totalPrice}`);
    return [shortestRoute, totalTime, totalPrice];
}

const findOptimalRoute = (originPoint, destinationPoint) => {
    const { nearestMode: originMode, nearestStation: originStation } = findNearestStationAnyMode(originPoint);
    const { nearestMode: destinationMode, nearestStation: destinationStation } = findNearestStationAnyMode(destinationPoint);

    const bestTimes = {};
    const bestPaths = {};

    for (const mode in stations) {
        for (const station in stations[mode]) {
            bestTimes[station] = Infinity;
        }
    }

    bestTimes[originStation] = 0;
    let queue = [[originStation, 0]];

    while (queue.length > 0) {
        queue.sort((a, b) => a[1] - b[1]);
        const [currentStation, currentTime] = queue.shift();

        for (const mode in routes) {
            for (const routeId of getRoutesThroughStation(mode, currentStation)) {
                for (const nextStation of routes[mode][routeId]) {
                    const travelTime = 1; // 1 tick per stop
                    const arrivalTime = currentTime + travelTime;

                    if (arrivalTime < bestTimes[nextStation]) {
                        bestTimes[nextStation] = arrivalTime;
                        bestPaths[nextStation] = currentStation;
                        queue.push([nextStation, arrivalTime]);
                    }
                }
            }
        }

        for (const [nextStation, transferTime] of getTransfersFromStation(currentStation)) {
            const arrivalTime = currentTime + transferTime;
            if (arrivalTime < bestTimes[nextStation]) {
                bestTimes[nextStation] = arrivalTime;
                bestPaths[nextStation] = currentStation;
                queue.push([nextStation, arrivalTime]);
            }
        }
    }

    const path = [];
    let current = destinationStation;
    while (current !== originStation) {
        path.push(current);
        current = bestPaths[current];
        if (current === undefined) {
            console.log(`Path not found from ${originStation} to ${destinationStation}`);
            return null;
        }
    }
    path.push(originStation);
    path.reverse();
    console.log(`Path for public transport ${JSON.stringify(path)}`);
    return path;
}

const buildDetailedItinerary = (optimalPath, originPoint, destinationPoint) => {
    const detailedItinerary = [];

    if (!optimalPath) {
        console.log("Optimal path is empty, returning empty itinerary.");
        return detailedItinerary;
    }

    if (optimalPath.length === 1) {
        console.log(`Optimal path has only one station: ${optimalPath}. No public transport needed.`);
        return detailedItinerary;
    }

    const { nearestMode: originStationMode, nearestStation: originStation } = findNearestStationAnyMode(originPoint);
    detailedItinerary.push(['to station', [originPoint, originStation]]);

    for (let i = 0; i < optimalPath.length - 1; i++) {
        const currentStation = optimalPath[i];
        const nextStation = optimalPath[i + 1];

        const currentMode = currentStation.startsWith('B') ? 'bus' : 'train';
        const nextMode = nextStation.startsWith('B') ? 'bus' : 'train';

        if (currentMode === nextMode) {
            for (const [routeId, stationSequence] of Object.entries(routes[currentMode])) {
                if (stationSequence.includes(currentStation) && stationSequence.includes(nextStation)) {
                    if (detailedItinerary[detailedItinerary.length - 1][0] === 'transfer' ||
                        detailedItinerary[detailedItinerary.length - 1][0] !== currentMode) {
                        detailedItinerary.push([currentMode, routeId, [currentStation]]);
                    }
                    detailedItinerary[detailedItinerary.length - 1][2].push(nextStation);
                    break;
                }
            }
        } else {
            detailedItinerary.push(['transfer', [currentStation, nextStation]]);
        }
    }

    detailedItinerary.push(['to destination', [optimalPath[optimalPath.length - 1], destinationPoint]]);
    return detailedItinerary;
}

const calculateTotalTimeAndPricePublic = (detailedItinerary, walkingSpeed, busStopSpeed, trainStopSpeed, busStopPrice, trainStopPrice) => {
    let totalTime = 0;
    let totalPrice = 0;

    for (const segment of detailedItinerary) {
        if (segment[0] === 'to station') {
            const getOnStationName = segment[1][1];
            const getOnStationCoordinates = getOnStationName.startsWith('T') ? stations['train'][getOnStationName] :
                stations['bus'][getOnStationName];

            const walkingRoute = dijkstraWithoutDiagonals(segment[1][0], getOnStationCoordinates);
            const walkDistance = calculateTotalRouteLength(walkingRoute);
            const walkTime = walkDistance / walkingSpeed;
            totalTime += walkTime;
        } else if (segment[0] === 'bus' || segment[0] === 'train') {
            const mode = segment[0];
            const routeId = segment[1];
            const stops = segment[2];
            const routeList = routes[mode][routeId];

            const getOnStop = stops[0];
            const getOffStop = stops[stops.length - 1];
            const getOnIndex = routeList.indexOf(getOnStop);
            const getOffIndex = routeList.indexOf(getOffStop);
            const numberOfStops = getOffIndex - getOnIndex + 1;

            const eachStopSpeed = mode === 'bus' ? busStopSpeed : trainStopSpeed;
            const pricePerStop = mode === 'bus' ? busStopPrice : trainStopPrice;
            const travelTime = numberOfStops * eachStopSpeed;
            const waitingTime = 1; // Default waiting time is 1 step
            const segmentTime = travelTime + waitingTime;

            totalTime += segmentTime;

            const segmentPrice = numberOfStops * parseFloat(pricePerStop);
            totalPrice += segmentPrice;
        } else if (segment[0] === 'transfer') {
            const transferStations = segment[1];
            const transferTime = transfers[transferStations];
            if (transferTime === undefined) {
                throw new Error(`Transfer from ${transferStations[0]} to ${transferStations[1]} is not recognized.`);
            }
            totalTime += transferTime;
        }
    }
    console.log(`Test calculate_total_time_and_price_public: ${totalTime}, ${totalPrice}`);
    return [totalTime, totalPrice];
}

const findNearestStationAnyMode = (point) => {
    let minDistance = Infinity;
    let nearestStation = null;
    let nearestMode = null;

    for (const mode in stations) {
        for (const [stationId, stationPoint] of Object.entries(stations[mode])) {
            const distance = Math.hypot(point[0] - stationPoint[0], point[1] - stationPoint[1]);
            if (distance < minDistance) {
                minDistance = distance;
                nearestStation = stationId;
                nearestMode = mode;
            }
        }
    }
    return { nearestMode, nearestStation };
}

const getRoutesThroughStation = (mode, station) => {
    return Object.keys(routes[mode]).filter(routeId => routes[mode][routeId].includes(station));
}

const getTransfersFromStation = (station) => {
    return Object.entries(transfers).filter(([key, _]) => key.split(',')[0] === station)
        .map(([key, time]) => [key.split(',')[1], time]);
}

const stations = {
    train: {
        'T1': [0, 0],
        'T2': [2, 2],
        'T3': [5, 4],
        'T4': [8, 4],
        'T5': [9, 5],
        'T6': [12, 6]
    },
    bus: {
        'B1': [1, 5],
        'B2': [2, 4],
        'B3': [2, 3],
        'B4': [3, 2],
        'B5': [3, 1],
        'B6': [4, 1],
        'B7': [4, 0],
        'B8': [5, 0],
        'B9': [6, 0],
        'B10': [7, 0],
        'B11': [12, 0],
        'B12': [10, 1],
        'B13': [8, 2],
        'B14': [7, 3],
        'B15': [7, 5],
        'B16': [6, 6],
        'B17': [4, 6],
        'B21': [13, 1],
        'B22': [12, 2],
        'B23': [11, 4],
        'B24': [9, 6],
        'B25': [9, 7]
    }
};

const routes = {
    train: {
        'RT1': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
    },
    bus: {
        'RB1': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10'],
        'RB2': ['B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17'],
        'RB3': ['B21', 'B22', 'B23', 'B24', 'B25']
    }
};

const transfers = {
    'T2,B3': 1, // Transfer time between station T2 and B3 is 1 step
    'T2,B4': 1,
    'B4,T2': 1,
    'B3,T2': 1,
    'T4,B15': 2,
    'B15,T4': 2,
    'T4,B14': 2,
    'B14,T4': 2,
    'T5,B24': 2,
    'B24,T5': 2,
    'T9,B28': 1,
    'B28,T9': 2
};

// Test the findOptimalRoute function
const optimalPath = findOptimalRoute([0, 0], [13, 3]);
const detailedItinerary = buildDetailedItinerary(optimalPath, [0, 0], [13, 3]);
console.log(`detailedItinerary is ${JSON.stringify(detailedItinerary)}`);
const [totalTime, totalPrice] = calculateTotalTimeAndPricePublic(detailedItinerary, 1, 3, 3, 1.5, 1.5);
console.log(`Test calculate_total_time_and_price_public: ${totalTime}, ${totalPrice}`);
