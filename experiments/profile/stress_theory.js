// Define the base processing times for each step
const processingTimes = {
    queryRewrite: 2.45, // average of 2.1-2.8s
    hyde: 4.2,
    retrieveRerank: 4.0, // baseline for a single query
    subQueryAnswer: 4.7,
    finalAnswer: 1.7
  };
  
  // Define the distribution of query types
  const queryDistribution = {
    singleSubQuery: 0.80,
    twoSubQueries: 0.15,
    threeSubQueries: 0.05
  };
  
  // Function to simulate the processing time for a single request
  function simulateRequest() {
    // Determine number of sub-queries based on distribution
    let rand = Math.random();
    let numSubQueries;
    if (rand < queryDistribution.singleSubQuery) {
      numSubQueries = 1;
    } else if (rand < queryDistribution.singleSubQuery + queryDistribution.twoSubQueries) {
      numSubQueries = 2;
    } else {
      numSubQueries = 3;
    }
    
    // Calculate time for each step
    const queryRewriteTime = processingTimes.queryRewrite;
    const hydeTime = processingTimes.hyde;
    
    // For step 4, this is only needed for multiple sub-queries
    const subQueryAnswerTime = numSubQueries > 1 ? processingTimes.subQueryAnswer : 0;
    
    const finalAnswerTime = processingTimes.finalAnswer;
    
    return {
      numSubQueries,
      queryRewriteTime,
      hydeTime,
      subQueryAnswerTime,
      finalAnswerTime
    };
  }
  
  // Function to calculate total latency for a request with concurrent users
  function calculateLatency(request, concurrentUsers) {
    // Retrieve and rerank step is affected by concurrent users
    // The processing effort is divided, so time increases linearly with users
    const retrieveRerankTime = processingTimes.retrieveRerank * request.numSubQueries * concurrentUsers;
    
    // Total latency is the sum of all step times
    // Steps 1, 2, 4, 5 are handled by 3rd party and not affected by concurrent users
    return request.queryRewriteTime + request.hydeTime + retrieveRerankTime + 
           request.subQueryAnswerTime + request.finalAnswerTime;
  }
  
  // Function to run simulations and calculate percentiles
  function runSimulations(numUsers, numSimulations = 10000) {
    const latencies = [];
    
    for (let i = 0; i < numSimulations; i++) {
      const requests = [];
      for (let j = 0; j < numUsers; j++) {
        requests.push(simulateRequest());
      }
      
      // Calculate latency for each request
      const requestLatencies = requests.map(req => calculateLatency(req, numUsers));
      latencies.push(...requestLatencies);
    }
    
    // Sort latencies to calculate percentiles
    latencies.sort((a, b) => a - b);
    
    const p50 = latencies[Math.floor(latencies.length * 0.5)];
    const p95 = latencies[Math.floor(latencies.length * 0.95)];
    const p99 = latencies[Math.floor(latencies.length * 0.99)];
    
    return { p50, p95, p99 };
  }
  
  // Run analysis for 1, 5, and 10 users
  const results = {};
  [1, 3, 5, 10].forEach(numUsers => {
    results[numUsers] = runSimulations(numUsers);
  });
  
  console.log("Latency Analysis Results (in seconds):");
  console.log(JSON.stringify(results, null, 2));
  
  // Let's also analyze the distribution of query types and their individual latencies
  function analyzeQueryTypes() {
    const singleSubQuery = {
      steps: {
        queryRewrite: processingTimes.queryRewrite,
        hyde: processingTimes.hyde,
        retrieveRerank: processingTimes.retrieveRerank,
        finalAnswer: processingTimes.finalAnswer
      }
    };
    singleSubQuery.total = Object.values(singleSubQuery.steps).reduce((a, b) => a + b, 0);
    
    const twoSubQueries = {
      steps: {
        queryRewrite: processingTimes.queryRewrite,
        hyde: processingTimes.hyde,
        retrieveRerank: processingTimes.retrieveRerank * 2,
        subQueryAnswer: processingTimes.subQueryAnswer,
        finalAnswer: processingTimes.finalAnswer
      }
    };
    twoSubQueries.total = Object.values(twoSubQueries.steps).reduce((a, b) => a + b, 0);
    
    const threeSubQueries = {
      steps: {
        queryRewrite: processingTimes.queryRewrite,
        hyde: processingTimes.hyde,
        retrieveRerank: processingTimes.retrieveRerank * 3,
        subQueryAnswer: processingTimes.subQueryAnswer,
        finalAnswer: processingTimes.finalAnswer
      }
    };
    threeSubQueries.total = Object.values(threeSubQueries.steps).reduce((a, b) => a + b, 0);
    
    return {
      singleSubQuery,
      twoSubQueries,
      threeSubQueries
    };
  }
  
  const queryTypeAnalysis = analyzeQueryTypes();
  console.log("\nBase Latency by Query Type (1 user):");
  console.log(JSON.stringify(queryTypeAnalysis, null, 2));
