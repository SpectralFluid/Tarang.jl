# Command-line parsing for merge_netcdf.jl.
#
# This file has no package dependencies, so the merge script runs in Tarang's own
# environment and the parser can be tested without loading Tarang or merging files.

# Long option name => default value. Boolean defaults mark flags; every other
# option takes a value, written `--name=VALUE` or `--name VALUE`.
const MERGE_OPTION_DEFAULTS = Dict{String, Any}(
    "set" => 0,
    "output" => "",
    "mode" => "reconstruct",
    "cleanup" => false,
    "auto" => false,
    "quiet" => false,
    "help" => false,
    "version" => false,
)

const MERGE_SHORT_OPTIONS = Dict{Char, String}(
    's' => "set", 'o' => "output", 'm' => "mode",
    'c' => "cleanup", 'a' => "auto", 'q' => "quiet",
    'h' => "help", 'v' => "version",
)

"""
    parse_arguments(argv=ARGS) -> Dict{String, Any}

Parse the merge script's command line into a dictionary holding every option in
`MERGE_OPTION_DEFAULTS` plus `"handlers"`, the positional handler names.

Short flags may be grouped (`-cq`); a short value option takes the rest of its
token (`-s2`) or the next argument (`-s 2`). Everything after `--` is a handler
name. Unknown options, missing values and a non-integer `--set` throw
`ArgumentError`.
"""
function parse_arguments(argv::AbstractVector{<:AbstractString}=ARGS)
    args = copy(MERGE_OPTION_DEFAULTS)
    handlers = String[]
    i = 1
    while i <= length(argv)
        token = argv[i]
        if token == "--"
            append!(handlers, argv[(i + 1):end])
            break
        elseif startswith(token, "--")
            name, value = occursin('=', token) ? split(token[3:end], '='; limit=2) :
                                                 (token[3:end], nothing)
            haskey(MERGE_OPTION_DEFAULTS, name) ||
                throw(ArgumentError("unrecognized option --$name"))
            if MERGE_OPTION_DEFAULTS[name] isa Bool
                value === nothing || throw(ArgumentError("option --$name takes no value"))
                args[name] = true
            else
                if value === nothing
                    i < length(argv) || throw(ArgumentError("option --$name requires a value"))
                    i += 1
                    value = argv[i]
                end
                args[name] = _merge_option_value(name, value)
            end
        elseif startswith(token, "-") && length(token) > 1
            k = nextind(token, 1)
            while k <= lastindex(token)
                flag = token[k]
                name = get(MERGE_SHORT_OPTIONS, flag, nothing)
                name === nothing && throw(ArgumentError("unrecognized option -$flag"))
                if MERGE_OPTION_DEFAULTS[name] isa Bool
                    args[name] = true
                    k = nextind(token, k)
                    continue
                end
                value = token[nextind(token, k):end]
                if isempty(value)
                    i < length(argv) || throw(ArgumentError("option -$flag requires a value"))
                    i += 1
                    value = argv[i]
                end
                args[name] = _merge_option_value(name, value)
                break
            end
        else
            push!(handlers, token)
        end
        i += 1
    end
    args["handlers"] = handlers
    return args
end

function _merge_option_value(name::AbstractString, value::AbstractString)
    MERGE_OPTION_DEFAULTS[name] isa Int || return String(value)
    parsed = tryparse(Int, value)
    parsed === nothing &&
        throw(ArgumentError("option --$name expects an integer, got \"$value\""))
    return parsed
end
