#! /usr/bin/env zsh

# To collect subissues:
#
# $ get_subissues <ISSUE NUMBER> number
# $ for issue in ${REPLY[@]}; do
# $     issue_to_markdown $issue
# $ done
#
# That will create markdown files under .issues/

function get_subissues() {
    # Check if gh is installed
    if ! command -v gh > /dev/null 2>&1; then
        echo "gh is not installed"
        return 1
    fi

    # Check if gh is logged in
    if ! gh auth status > /dev/null 2>&1; then
        echo "gh is not logged in"
        return 1
    fi

    # Check if the issue number is provided
    if [ -z "$1" ]; then
        echo "Usage: ghissue <issue_number>"
        return 1
    fi

    # Mode should be one of
    # - raw: The raw JSON is returned
    # - json: The whole JSON is returned (minimized, default)
    # - number: List of issue numbers
    # - url: List of issue URLs
    local mode="${2:-json}"
    if [ "$mode" != "raw" ] && [ "$mode" != "json" ] && [ "$mode" != "number" ] && [ "$mode" != "url" ]; then
        echo "Invalid mode: $mode"
        return 1
    fi

    local issue_num="$1"
    local parent_id=$(gh issue view "${issue_num}" --json id -q .id)

    # Get the issue and the subissues
    sub_issues=$(gh api graphql \
    --paginate \
    -H GraphQL-Features:sub_issues \
    -H GraphQL-Features:issue_types \
    -f issueId="$parent_id" \
    -f query='
    query($issueId: ID!, $endCursor: String) {
        node(id: $issueId) {
            ... on Issue {
                subIssues(first: 100, after: $endCursor) {
                    totalCount
                    nodes {
                        title
                        number
                        url
                        closed
                    }
                }
            }
        }
    }
    ')

    # Check if the subissues are empty
    if [ -z "$sub_issues" ]; then
        echo "No subissues found"
        return 1
    fi

    # Return the right format, preferrably as a list
    local result
    case "$mode" in
        raw)
            echo "$sub_issues"
            return 0
            ;;
        json)
            result=$(echo "$sub_issues" | jq -rcM '.data.node.subIssues.nodes[]')
            ;;
        number)
            result=$(echo "$sub_issues" | jq -r '.data.node.subIssues.nodes[] | .number')
            ;;
        url)
            result=$(echo "$sub_issues" | jq -r '.data.node.subIssues.nodes[] | .url')
            ;;
        *)
            echo "Invalid mode: $mode"
            return 1
    esac
    # json_results=(${(f)"$(get_subissues 15 json)"})
    # declare -a listed_result=(${(f)"${result}"})
    # print ("${listed_result[@]}")
    local issues=(${(f)"${result}"})
    REPLY=("${issues[@]}")
}


function issue_to_markdown_file() {
    # Check if gh is installed
    if ! command -v gh > /dev/null 2>&1; then
        echo "gh is not installed"
        return 1
    fi

    # Check if gh is logged in
    if ! gh auth status > /dev/null 2>&1; then
        echo "gh is not logged in"
        return 1
    fi

    # Check if the issue number is provided
    if [ -z "$1" ]; then
        echo "Usage: ghissue <issue_number>"
        return 1
    fi
    local issue_number="$1"

    # Get the filename to write to
    local filename="${2:-.issues/issue-${issue_number}.md}"

    # Check if the file or directory with that name exists
    if [ -e "$filename" ]; then
        echo "File or directory with that name already exists"
        return 1
    fi

    # Get the issue
    local issue="$(gh issue view "$issue_number" --json url,title,body,closed)"

    # Check if the issue is empty
    if [ -z "$issue" ]; then
        echo "No issue found"
        return 1
    fi


    local issue_title=$(jq -r '.title' <<< "$issue")
    local issue_body=$(jq -r '.body' <<< "$issue")
    local issue_closed=$(jq -r '.closed' <<< "$issue")
    local issue_url=$(jq -r '.url' <<< "$issue")

    # Check if the issue is closed
    if [ "$issue_closed" = "true" ]; then
        echo "Issue is closed"
        return 1
    fi

    # Write the issue to the file
    echo "Writing to $filename"
    echo "# ${issue_title}" > "$filename"
    echo "" >> "$filename"
    echo "$issue_body" >> "$filename"
    echo "" >> "$filename"
    echo "---" >> "$filename"
    echo "" >> "$filename"
    echo "[#${issue_number}](${issue_url})" >> "$filename"
}

function issue_to_markdown() {
    # Converts all the issue numbers to markdown files
    # Usage: issue_to_markdown <number> [<number> ...]
    for issue_number in "$@"; do
        issue_to_markdown_file "$issue_number"
    done
}
