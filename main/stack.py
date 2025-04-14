from prompt.system_prompt import verification_prompt,process_identification_prompt_v1,transactions_prompt,response_recent_transaction,response_verification
from prompt. user_prompt import response_recent_transaction_input,transactions_input_prompt,process_identification_prompt_input,verification_prompt_input

from components.tools.verifiaction import verify_user
from components.tools.execute import get_data

prompt_stack={
	"verification":{
		"system":verification_prompt,
		"input":verification_prompt_input},
	"process_identifiaction":{
		"system":process_identification_prompt_v1,
		"input":process_identification_prompt_input
	},
	"Recent_transactions":{
		"system":transactions_prompt,
		"input":transactions_input_prompt
	},
	"Response Generator":{
		"Recent_transactions":{
		"system":response_recent_transaction,
		"input":response_recent_transaction_input
	},
		"verification":{
			"system":response_verification,
			"input":response_recent_transaction_input
		}
	}
}

tool_stack={
	"verification":verify_user,
	"transactions":get_data
}