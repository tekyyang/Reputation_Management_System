# An example to get the remaining rate limit using the Github GraphQL API.

import requests
import json

# The GraphQL query (with a few aditional bits included) itself defined as a multi-line string.
def run_graphql_query(previous_cuscor):
    headers = {"Authorization": "token 59439ca09a4f8cb1543cfa5af7167961e57a8221"}

    def run_query(query):  # A simple function to use requests.post to make the API call. Note the json= section.
        request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)
        if request.status_code == 200:
            return request.json()
        else:
            raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))

    query = '''
    query{
      rateLimit(dryRun:false)
      {
        cost
        limit
        remaining
      }
      organization(login: "Shopify") {
        repository(name: "ios") {
          name
          pullRequests(first: 1,after:"%s" , orderBy:{field:CREATED_AT, direction:ASC}) {
            edges {
                      cursor
                      node {
                        id
                        state
                        title
                        bodyText
                        createdAt
                        lastEditedAt
                        closedAt
                        mergedAt
                        resourcePath
                        
                        labels(first:20){
                          nodes{
                            name
                            description
                          }
                        }
                        
                        mergedBy{
                          login
                                     }
                        changedFiles 
                       
                        editor { #The actor who edited this pull request's body
                                login
                       }
                        
                        participants(first:20){ #A list of Users that are participating in the Pull Request conversation. Like a superset of reviewer and commenters.
                                    nodes{
                                    login
                                     }
                        }
                        
                        reviews(first:20){
                            nodes{
                                author {login}
                                bodyText
                                  }
                        }    
            
                        comments(first:50) {
                          edges {
                            node {
                              author {login}
                              bodyText
                            }
                          }
                        }
                        
                        assignees(first:20){
                          nodes{
                            name
                            createdAt
                          }
                        }
                            }}}}}}'''%(previous_cuscor)
    result = run_query(query)  # Execute the query
    result_clean = {}
    result_clean['cursor'] = result['data']['organization']['repository']['pullRequests']['edges'][0]['cursor']
    result_clean['id'] = result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['id']
    result_clean['state'] = result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['state']
    result_clean['title'] = result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['title']
    result_clean['bodyText'] = result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['bodyText']
    result_clean['createdAt'] = result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['createdAt']
    result_clean['lastEditedAt'] = result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['lastEditedAt']
    result_clean['closedAt'] = result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['closedAt']
    result_clean['mergedAt'] = result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['mergedAt']
    try:
        result_clean['mergedBy'] = result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['mergedBy']['login']
    except:
        result_clean['mergedBy'] = None
    result_clean['changedFiles'] = result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['changedFiles']
    result_clean['editor'] = result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['editor']
    result_clean['participants'] = [i['login'] for i in result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['participants']['nodes']]
    result_clean['assignees'] = [i['name'] for i in result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['assignees']['nodes']]
    result_clean['reviewers'] = [i['author']['login'] for i in result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['reviews']['nodes']]
    result_clean['reviews'] = [i['bodyText'] for i in result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['reviews']['nodes']]
    result_clean['commenters'] = [i['node']['author']['login'] for i in result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['comments']['edges']]
    result_clean['comments'] = [i['node']['bodyText'] for i in result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['comments']['edges']]
    result_clean['resourcePath'] = result['data']['organization']['repository']['pullRequests']['edges'][0]['node']['resourcePath']

    remaining_rate_limit = result["data"]["rateLimit"]["remaining"]  # Drill down the dictionary
    # print("Remaining rate limit - {}".format(remaining_rate_limit))
    with open('/Users/yibingyang/Documents/5900X/ios_pr_test.json', 'a')as f:
        f.write(json.dumps(result_clean) + '\n')
        f.close()
    # print result_clean
    last_cursor = result['data']['organization']['repository']['pullRequests']['edges'][-1]['cursor']
    return last_cursor

n = 1
# cursor_1 = 'Y3Vyc29yOnYyOpK5MjAxNy0wNS0wMVQxOTozMzoxMC0wNDowMM4HD5Fr'
cursor_1 = 'Y3Vyc29yOnYyOpK5MjAxNy0wNS0xMFQxNDo1MToxMS0wNDowMM4HJlei'
for i in range(1,2000):
    if i == 1:
        cursor_2 =run_graphql_query(cursor_1)
        print n
    else:
        locals()['cursor' + '_' + str(i + 1)] = run_graphql_query(locals()['cursor' + '_' + str(i)])
        n+=1
        print n

