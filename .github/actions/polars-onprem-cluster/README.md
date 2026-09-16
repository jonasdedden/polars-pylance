# Polars on-premises cluster in kind

A composite action that stands up a real Polars cluster on a GitHub-hosted
runner — a scheduler, workers and an S3-compatible object store in a `kind`
cluster — and exposes it on `localhost`, so a `polars-cloud` client in the same
job can run queries against it. A companion `teardown` action collects
diagnostics and uninstalls the release.

This repository uses it in `.github/workflows/polars-cloud-k8s.yml` to run
`tests/k8s/` against a cluster whose workers carry the commit under test.

## Usage

```yaml
jobs:
  e2e:
    runs-on: ubuntu-latest
    # A workspace allows one concurrent cluster on the free tier.
    concurrency: polars-onprem-cluster
    steps:
      - uses: actions/checkout@v7

      - id: cluster
        uses: ./.github/actions/polars-onprem-cluster
        with:
          client-id: ${{ secrets.POLARS_CLIENT_ID }}
          client-secret: ${{ secrets.POLARS_CLIENT_SECRET }}
          workspace-id: ${{ secrets.POLARS_WORKSPACE_ID }}
          python-version: "3.12.14"
          worker-requirements: |
            pylance
            polars-pylance

      - run: pytest tests/e2e
        env:
          POLARS_SCHEDULER_URI: ${{ steps.cluster.outputs.scheduler-uri }}
          POLARS_OBSERVATORY_URI: ${{ steps.cluster.outputs.observatory-uri }}
          S3_ENDPOINT: ${{ steps.cluster.outputs.object-store-endpoint }}

      - if: always()
        uses: ./.github/actions/polars-onprem-cluster/teardown
        with:
          collect-diagnostics: ${{ job.status == 'failure' }}
```

and on the client side:

```python
import os

import polars_cloud as pc

ctx = pc.ClusterContext(
    uri=os.environ["POLARS_SCHEDULER_URI"],
    observatory=pc.ClientOptions(uri=os.environ["POLARS_OBSERVATORY_URI"]),
)
lf.remote(ctx).distributed().collect()
```

## Inputs

| Input | Default | What it does |
| --- | --- | --- |
| `client-id`, `client-secret`, `workspace-id` | required | Service account of a Kubernetes-type workspace (see [Credentials](#credentials)) |
| `cluster-id` | derived from the run | UUID the cluster registers under |
| `chart-version` | `3.0.2` | `polars-inc/polars` chart version |
| `python-version` | `3.12.14` | Worker runtime image; must equal the client's Python, patch included |
| `worker-requirements` | empty | requirements.txt installed on every worker at boot |
| `worker-replicas` | `2` | Number of worker pods |
| `values` | empty | Extra Helm values as inline YAML, applied last |
| `object-store` | `true` | Deploy SeaweedFS as `http://objectstore:8333` and route anonymous results through it |
| `object-store-access-key-id`, `object-store-secret-access-key` | throwaway values | Credentials of that store |
| `run-chart-tests` | `true` | Run the chart's own `helm test` suite as a readiness check |
| `kind-cluster-name` | `polars` | Name of the kind cluster |
| `kind-config` | bundled | kind config; a replacement must keep the host port mappings |
| `namespace`, `release` | `polars` | Where the chart is installed |

## Outputs

| Output | Value |
| --- | --- |
| `scheduler-uri` | `http://localhost:5051` |
| `observatory-uri` | `http://localhost:3001` |
| `object-store-endpoint` | `http://objectstore:8333`, or empty without a store |
| `object-store-bucket` | `polars`, or empty without a store |
| `cluster-id`, `namespace`, `release` | as deployed |

The `teardown` action takes `namespace`, `release`, `collect-diagnostics`,
`artifact-name` and `retention-days`.

## How it fits together

| File | What it is |
| --- | --- |
| `action.yml` | kind cluster, secrets, object store, chart install, NodePort pinning, endpoint wait, `helm test` |
| `teardown/action.yml` | nodes, resources, events and pod logs as an artifact; `helm uninstall` |
| `kind-config.yaml` | a control plane and two workers; NodePorts mapped to host ports, so the client needs no port-forward |
| `manifests/object-store.yaml` | SeaweedFS in `mini` mode with a `polars` bucket |
| `values/cluster.yaml` | chart values sized for a 4 vCPU / 16 GiB runner |
| `values/object-store.yaml` | anonymous query results in that store |

**One hostname for the object store.** Workers reach the store through its
in-cluster Service, and the runner through a NodePort plus an `/etc/hosts` entry,
both as `objectstore:8333`. One `storage_options` dict therefore works on both
sides, and presigned result URLs the scheduler hands out resolve on the runner.

**What the workers can import.** A query that ships Python — an IO source, a
`sink_batches` callback pickled by reference — can only run on a worker that
has the package installed, so `worker-requirements` decides which code the
cluster tests. Pin `polars` to the client's version: polars-cloud rejects a
mismatch. To test unreleased code, point a line at a GitHub archive of the
commit, e.g. `mypackage @ https://github.com/owner/repo/archive/<sha>.tar.gz`.
Workers install it over the internet at boot, which makes the first start slow.

## Credentials

On-premises clusters still authenticate against the Polars Cloud control plane:

1. Sign in at [cloud.pola.rs](https://cloud.pola.rs) and create a workspace
   with **Kubernetes** as the deployment type.
2. Create a service account in the workspace settings and copy the client ID
   and secret straight away — they are not shown again. The workspace ID is on
   the same page, or from `pc workspace list`.
3. Add them as repository secrets, e.g. `POLARS_CLIENT_ID`,
   `POLARS_CLIENT_SECRET` and `POLARS_WORKSPACE_ID`.

Pull requests from forks get no secrets; check for them first and skip the job,
as this repository's workflow does, rather than letting it fail.

## Things to know

- **One cluster at a time.** The free tier allows a single concurrent cluster
  per workspace, so give every job using the workspace one `concurrency` group,
  and always run `teardown`: uninstalling is what hands the slot back.
- **Client and cluster versions have to match.** `polars-cloud==0.11.2` requires
  `polars==1.44.2`. Chart 3.0.2 ships its in-cluster tests against client
  0.11.1, which is not on PyPI: if `helm test` passes but your client fails on a
  protocol or version error, that gap is the first thing to check.
- **Python versions have to match too.** Python IO sources carry closures that
  Polars cannot deserialize under a different Python version, patch release
  included. Set up the client's Python from the same `python-version`.
- **Worker memory.** Each worker gets 3 GiB, which is what a 4 vCPU / 16 GiB
  runner can spare for two. Override it through `values` on a larger runner.

## Running it locally

The action is a sequence of plain `kubectl` and `helm` commands, so the same
cluster can be built by hand:

```sh
ACTION=.github/actions/polars-onprem-cluster
kind create cluster --name polars --config "$ACTION/kind-config.yaml"
kubectl create namespace polars

kubectl -n polars create secret generic polars-onprem-license \
  --from-literal=client_id="$POLARS_CLIENT_ID" \
  --from-literal=client_secret="$POLARS_CLIENT_SECRET" \
  --from-literal=workspace_id="$POLARS_WORKSPACE_ID"
kubectl -n polars create secret generic objectstore-credentials \
  --from-literal=aws_access_key_id=ci-access-key \
  --from-literal=aws_secret_access_key=ci-secret-key-not-sensitive
kubectl -n polars apply -f "$ACTION/manifests/object-store.yaml"
echo "127.0.0.1 objectstore" | sudo tee -a /etc/hosts

printf 'pylance\npolars-pylance\n' > /tmp/requirements.txt
helm upgrade --install polars polars --repo https://polars-inc.github.io/helm-charts \
  --version 3.0.2 --namespace polars \
  --values "$ACTION/values/cluster.yaml" --values "$ACTION/values/object-store.yaml" \
  --set-string fullnameOverride=polars \
  --set-string clusterId="$(uuidgen)" \
  --set-string runtime.composed.runtime.tag=3.12.14-slim-bookworm \
  --set-file runtime.composed.requirements=/tmp/requirements.txt \
  --wait --timeout 15m
kubectl -n polars patch service polars-scheduler \
  -p '{"spec":{"ports":[{"port":5051,"nodePort":30051}]}}'
kubectl -n polars patch service polars-observatory \
  -p '{"spec":{"ports":[{"port":3001,"nodePort":30001}]}}'

# This repository's suite, then clean up to free the cluster slot.
POLARS_K8S_E2E=1 uv run --python 3.12.14 --group test --extra cloud \
  pytest tests/k8s -v -o filterwarnings=default
helm uninstall polars -n polars --wait
kind delete cluster --name polars
```
