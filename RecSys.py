from pyspark.mllib.recommendation import ALS, Rating
from pyspark import  SparkContext
import time
import statistics
import math
import sys

t1 = time.time()

# Load and parse the data
sc = SparkContext('local[*]', 'task2')
trainRDD = sc.textFile(sys.argv[1])
testRDD = sc.textFile(sys.argv[2])

trainRDD = trainRDD.map(lambda x:x.split(',')).filter(lambda x: x[0]!='user_id')
testRDD = testRDD.map(lambda x:x.split(',')).filter(lambda x: x[0]!= 'user_id')

def computeUserBasedCF(record123):
    if record123[0] not in userMap.keys() and record123[1] not in businessMap.keys():
        return (record123, 3.0)
    elif record123[0] not in userMap.keys():
        return (record123, avgbusinessMap[record123[1]])
    elif record123[1] not in businessMap.keys():
        return (record123, avguserMap[record123[0]])

    candidates = [(record123[0],i) for i in businessMap[record123[1]]]

    candidate_weights = []
    for record in candidates:
        user1 = userMap[record[0]]
        user2 = userMap[record[1]]
        weight = 0
        corated = set(user1.keys()).intersection(set(user2.keys()))

        corated_length = len(corated)

        if corated_length > 1:
            rating1 = []
            rating2 = []
            rating1_sum = 0
            rating2_sum = 0
            numerator = 0
            denominator_term1 = 0
            denominator_term2 = 0
            for i in corated:
                item1 = user1[i]
                item2 = user2[i]
                rating1.append(item1)
                rating2.append(item2)
                rating1_sum += item1
                rating2_sum += item2
            mean1 = rating1_sum / corated_length
            mean2 = rating2_sum / corated_length
            for i in range(corated_length):
                numerator += (rating1[i] - mean1) * (rating2[i] - mean2)
                denominator_term1 += math.pow(rating1[i] - mean1, 2)
                denominator_term2 += math.pow(rating2[i] - mean2, 2)
            denominator = math.sqrt(denominator_term1 * denominator_term2)
            if (denominator != 0):
                weight = numerator / denominator
        if weight > 0:
            candidate_weights.append((record, weight))

    numerator = 0
    denominator = 0
    for item in candidate_weights:
        sum = 0
        count = 0
        user_businessMap = userMap[item[0][1]]
        for i in user_businessMap:
            if i != record123[1]:
                sum += user_businessMap[i]
                count += 1
        mean = sum / count
        numerator += ((user_businessMap[record123[1]] - mean) * item[1])
        denominator += abs(item[1])

    if denominator != 0:
        second_term = numerator / denominator
    else:
        second_term = 0

    prediction = avguserMap[record123[0]] + second_term

    if prediction > 5 or prediction < 1:
        prediction = avguserMap[record123[0]]

    return (record123, prediction)

def computeItemBasedCF(record123):
    if record123[0] not in userMap.keys() and record123[1] not in businessMap.keys():
        return (record123, 3.0)
    elif record123[0] not in userMap.keys():
        return (record123, avgbusinessMap[record123[1]])
    elif record123[1] not in businessMap.keys():
        return (record123, avguserMap[record123[0]])

    candidates = [(record123[1],i) for i in userMap[record123[0]]]

    candidate_weights = []

    for record in candidates:
        business1 = businessMap[record[0]]
        business2 = businessMap[record[1]]
        corated = set(business1.keys()).intersection(set(business2.keys()))
        mean1 = avgbusinessMap[record[0]]
        mean2 = avgbusinessMap[record[1]]

        weight = 0
        numerator = 0
        denominator_term1 = 0
        denominator_term2 = 0

        if len(corated) < 30:
            continue

        for i in corated:
            numerator += (business1[i] - mean1) * (business2[i] - mean2)
            denominator_term1 += math.pow(business1[i] - mean1, 2)
            denominator_term2 += math.pow(business2[i] - mean2, 2)
        denominator = math.sqrt(denominator_term1 * denominator_term2)

        if (denominator != 0):
            weight = numerator / denominator

        if weight > 0:
            candidate_weights.append((record, weight))

    numerator = 0
    denominator = 0
    for item in candidate_weights:
        numerator += (businessMap[item[0][1]][record123[0]] * item[1])
        denominator += abs(item[1])

    if denominator != 0:
        prediction = numerator / denominator
    else:
        prediction = avgbusinessMap[record123[1]]

    if prediction > 5 or prediction < 1:
        prediction = avgbusinessMap[record123[1]]

    return (record123, prediction)


if sys.argv[3] == '1':
    unionRDD = trainRDD.union(testRDD)

    userMap = unionRDD.map(lambda x: (x[0], x[1])).groupByKey().zipWithIndex().map(lambda x: (x[0][0], x[1] + 1)).collectAsMap()
    businessMap = unionRDD.map(lambda x: (x[1], x[0])).groupByKey().zipWithIndex().map(lambda x: (x[0][0], x[1] + 1)).collectAsMap()

    userIDMap = {value: key for key, value in userMap.items()}
    businessIDMap = {value: key for key, value in businessMap.items()}

    ratings = trainRDD.map(lambda x: Rating(userMap[x[0]], businessMap[x[1]], float(x[2])))
    testing = testRDD.map(lambda x: Rating(userMap[x[0]], businessMap[x[1]], float(x[2])))

    # Build the recommendation model using Alternating Least Squares
    rank = 8
    numIterations = 8
    lmbda = 0.1

    model = ALS.train(ratings, rank, numIterations, lmbda)

    # Evaluate the model on training data
    testdata = testing.map(lambda x: (x[0], x[1]))
    predictedRDD = model.predictAll(testdata).map(lambda x: ((userIDMap[x[0]], businessIDMap[x[1]]), x[2]))

    testRDD = testRDD.map(lambda x:((x[0], x[1]), float(x[2])))
    extraRDD = testRDD.subtractByKey(predictedRDD).map(lambda x:(x[0], 3.0))

    predictedRDD = predictedRDD.union(extraRDD)

    # rmse = predictedRDD.join(testRDD).map(lambda x: math.pow(x[1][0] - x[1][1], 2)).mean()

    output_str = 'user_id, business_id, prediction\n'
    for item in predictedRDD.collect():
        output_str += str(item[0][0])+ ',' + str(item[0][1]) + ',' + str(item[1]) + '\n'

    f = open(sys.argv[4], "w")
    f.write(output_str)
    f.close()

    # print("rmse = ", math.sqrt(rmse))
    print("Time = ", time.time() - t1)

elif sys.argv[3] == '2':

    avguserMap = trainRDD.map(lambda x: (x[0], float(x[2]))).groupByKey().map(lambda x: (x[0], statistics.mean(x[1]))).collectAsMap()

    avgbusinessMap = trainRDD.map(lambda x: (x[1], float(x[2]))).groupByKey().map(lambda x: (x[0], statistics.mean(x[1]))).collectAsMap()

    # Businesses grouped by users who rated them
    businessMap = trainRDD.map(lambda x: (x[1], x[0])).groupByKey().collectAsMap()

    # Users grouped by businesses they rated along with the rating
    userMap = trainRDD.map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey().mapValues(lambda x: dict(x)).collectAsMap()

    computeRDD = testRDD.map(lambda x: (x[0], x[1]))

    predictedRDD = computeRDD.map(computeUserBasedCF)

    # testRDD = testRDD.map(lambda x: ((x[0], x[1]), float(x[2])))
    #
    # rmse = predictedRDD.join(testRDD).map(lambda x: math.pow(x[1][0] - x[1][1], 2)).mean()

    output_str = 'user_id, business_id, prediction\n'
    for item in predictedRDD.collect():
        output_str += item[0][0] + ',' + item[0][1] + ',' + str(item[1]) + '\n'

    f = open(sys.argv[4], "w")
    f.write(output_str)
    f.close()

    # print("rmse = ", math.sqrt(rmse))
    print("Time = ", time.time() - t1)

elif sys.argv[3] == '3':

    avguserMap = trainRDD.map(lambda x: (x[0], float(x[2]))).groupByKey().map(lambda x: (x[0], statistics.mean(x[1]))).collectAsMap()

    avgbusinessMap = trainRDD.map(lambda x: (x[1], float(x[2]))).groupByKey().map(lambda x: (x[0], statistics.mean(x[1]))).collectAsMap()

    # Users grouped by businesses they rated
    userMap = trainRDD.map(lambda x: (x[0], x[1])).groupByKey().collectAsMap()

    # Businesses grouped by users who rated it along with the rating
    businessMap = trainRDD.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(lambda x: dict(x)).collectAsMap()

    computeRDD = testRDD.map(lambda x: (x[0], x[1]))

    predictedRDD = computeRDD.map(computeItemBasedCF)

    # testRDD = testRDD.map(lambda x: ((x[0], x[1]), float(x[2])))
    #
    # rmse = predictedRDD.join(testRDD).map(lambda x: math.pow(x[1][0] - x[1][1], 2)).mean()


    output_str = 'user_id, business_id, prediction\n'
    for item in predictedRDD.collect():
        output_str += item[0][0] + ',' + item[0][1] + ',' + str(item[1]) + '\n'

    f = open(sys.argv[4], "w")
    f.write(output_str)
    f.close()

    # print("rmse = ", math.sqrt(rmse))
    print("Time = ", time.time() - t1)
