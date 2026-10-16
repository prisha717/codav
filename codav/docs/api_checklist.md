0. GET /health
It verifies that the backend service is up and running.
Expected response: 200 OK
Body: {"status": "ok"}

1. GET /datasets
It returns all of the datasets stored inside the database.
If datasets exist, Expected response: 200 OK
Body: List of the dataset objects
If no datasets exist, Expected response: 200 OK
Body: []

2. GET /datasets/{dataset_id}
The purpose is to return the metadata for a single dataset to support inspection before any visualisation.
Valid dataset_id, Expected response: 200 OK
Body: Dataset object
Invalid dataset_id, Expected response: 404 Not Found
Body: { "detail": "Dataset not found" }

3. GET /datasets/{dataset_id}/preview?limit=20
Return a small preview of dataset rows for UI exploration.
Expected response (valid dataset_id): 200 OK
Body:
{
  "dataset_id": 1,
  "limit": 20,
  "rows": [...]
}
Expected response (valid dataset_id but no preview rows stored yet): 200 OK
Body:
{
  "dataset_id": 1,
  "limit": 20,
  "rows": []
}
Expected response (invalid dataset_id): 404 Not Found
Body:
{ "detail": "Dataset not found" }
The limit parameter is constrained to prevent large payloads and protect performance.

4. POST /visualisations
Create and persist a visualisation linked to a dataset.
Expected response (valid dataset_id): 200 OK
Body: created visualisation object including generated id.
Expected response (invalid dataset_id): 404 Not Found
Body:
{ "detail": "Dataset not found" }
5. GET /visualisations?dataset_id={dataset_id}
Purpose: Return all visualisations associated with a dataset.
Expected response (valid dataset_id): 200 OK
Body: list of visualisations (may be empty).
Expected response (invalid dataset_id): 404 Not Found
Body:
{ "detail": "Dataset not found" }