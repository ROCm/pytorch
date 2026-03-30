#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="PVT_kwDOAULW6s4Ab2br"
SPRINT_FIELD_ID="PVTIF_lADOAULW6s4Ab2brzgR9L1s"
ASSIGNEE="ethanwee1"

get_iteration_ids() {
  local data
  data=$(gh api graphql -f query='
    {
      organization(login: "ROCm") {
        projectV2(number: 18) {
          field(name: "Sprint") {
            ... on ProjectV2IterationField {
              configuration {
                iterations { id title startDate duration }
                completedIterations { id title startDate duration }
              }
            }
          }
        }
      }
    }')

  local today
  today=$(date +%Y-%m-%d)

  local all_iterations
  all_iterations=$(echo "$data" | jq -r '
    [.data.organization.projectV2.field.configuration.iterations[],
     .data.organization.projectV2.field.configuration.completedIterations[]]
    | sort_by(.startDate)')

  CURRENT_ID=$(echo "$all_iterations" | jq -r --arg today "$today" '
    [.[] | select(.startDate <= $today)] | last | .id')
  CURRENT_TITLE=$(echo "$all_iterations" | jq -r --arg today "$today" '
    [.[] | select(.startDate <= $today)] | last | .title')

  PREV_ID=$(echo "$all_iterations" | jq -r --arg today "$today" '
    [.[] | select(.startDate <= $today)] | .[-2] | .id')
  PREV_TITLE=$(echo "$all_iterations" | jq -r --arg today "$today" '
    [.[] | select(.startDate <= $today)] | .[-2] | .title')

  echo "Previous sprint: $PREV_TITLE ($PREV_ID)"
  echo "Current sprint:  $CURRENT_TITLE ($CURRENT_ID)"
}

fetch_items() {
  local cursor=""
  local has_next=true
  ITEM_IDS=()

  echo ""
  echo "Fetching items assigned to $ASSIGNEE in '$PREV_TITLE'..."

  while [ "$has_next" = "true" ]; do
    local after_clause=""
    if [ -n "$cursor" ]; then
      after_clause=", after: \"$cursor\""
    fi

    local result
    result=$(gh api graphql -f query="
      {
        node(id: \"$PROJECT_ID\") {
          ... on ProjectV2 {
            items(first: 100$after_clause) {
              pageInfo { hasNextPage endCursor }
              nodes {
                id
                fieldValueByName(name: \"Sprint\") {
                  ... on ProjectV2ItemFieldIterationValue {
                    iterationId
                  }
                }
                content {
                  ... on Issue {
                    title
                    number
                    assignees(first: 10) {
                      nodes { login }
                    }
                  }
                  ... on PullRequest {
                    title
                    number
                    assignees(first: 10) {
                      nodes { login }
                    }
                  }
                }
              }
            }
          }
        }
      }")

    local matches
    matches=$(echo "$result" | jq -r --arg prev "$PREV_ID" --arg assignee "$ASSIGNEE" '
      .data.node.items.nodes[]
      | select(
          .fieldValueByName.iterationId == $prev
          and (.content.assignees.nodes // [] | map(.login) | index($assignee))
        )
      | "\(.id)\t#\(.content.number // "?")\t\(.content.title // "unknown")"')

    if [ -n "$matches" ]; then
      while IFS=$'\t' read -r item_id number title; do
        ITEM_IDS+=("$item_id")
        echo "  Found: $number - $title"
      done <<< "$matches"
    fi

    has_next=$(echo "$result" | jq -r '.data.node.items.pageInfo.hasNextPage')
    cursor=$(echo "$result" | jq -r '.data.node.items.pageInfo.endCursor')
  done

  echo ""
  echo "Found ${#ITEM_IDS[@]} item(s) to move."
}

move_items() {
  if [ ${#ITEM_IDS[@]} -eq 0 ]; then
    echo "Nothing to do."
    return
  fi

  echo ""
  read -rp "Move all ${#ITEM_IDS[@]} item(s) from '$PREV_TITLE' to '$CURRENT_TITLE'? [y/N] " confirm
  if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    return
  fi

  echo ""
  for item_id in "${ITEM_IDS[@]}"; do
    echo -n "  Moving $item_id... "
    gh api graphql -f query="
      mutation {
        updateProjectV2ItemFieldValue(input: {
          projectId: \"$PROJECT_ID\"
          itemId: \"$item_id\"
          fieldId: \"$SPRINT_FIELD_ID\"
          value: { iterationId: \"$CURRENT_ID\" }
        }) {
          projectV2Item { id }
        }
      }" > /dev/null
    echo "done"
  done

  echo ""
  echo "Moved ${#ITEM_IDS[@]} item(s) to '$CURRENT_TITLE'."
}

get_iteration_ids
fetch_items
move_items
