"""
    Verifier Workflow:
    Stage 1: Completeness checker检查当前步骤是否完备，如果完备进入stage 2
    Stage 2: NL-verifier用自然语言分析并验证当前步骤的正确性，如果发现其中包含复杂逻辑或计算进入stage 3
    Stage 3: NL-verifier Deliberate thinking / CI-verifier基于代码校验 
"""

from typing import Callable, Dict, List, Union
from pydantic import BaseModel, Field

from lagent.llms import BaseLLM
from lagent.agents import MathCoder
from lagent.prompts.parsers import StrParser, ToolParser, KeyParser
from lagent.agents.aggregator import InternLMToolAggregator
from lagent.agents.agent import Agent
from lagent.schema import AgentMessage, AgentStatusCode


complete_check_sys = (
    "Given a math problem and its corresponding multi-step solution, "
    "analyze whether a specified step is sufficiently complete to allow for verification. "
    "First, assess the specific progress made in that step, "
    "and then provide a judgment on its completeness for validation purposes and return **COMPLETE** or **INCOMPLETE**"
)

complete_check_usr = "### Problem\n\n{problem}\n\n### Previous steps\n\n{previous_steps}\n\n### Current step\n\n{current_step}\n\n### Following steps\n\n{following_steps}"

nl_ver_sys = (
    "First, analyze the progress of the current step along with the calculations and logical operations involved. "
    "If the step is straightforward and easy to understand, directly verify its correctness and return **RIGHT** or **WRONG**. "
    "If the step involves complex calculations or logic that would be better verified through Python programming, return **NEED FURTHER VERIFICATION**."
)

nl_ver_usr = "### Problem\n\n{problem}\n\n### Reasoning history\n\n{reasoning_history}\n\n### Current step\n\n{current_step}"

ci_ver_sys = """## Task: Code-integrated Verification on Math Problem Solutions

## Annotator Profile:

- **Name:** Kronos
- **Languages:** Fluent in English and Chinese
- **Role Description:** As an expert verifier in math and programming, your task is to verify the correctness of each step in a math problem solution with the assistance of Python programming.  

## Input Format

The input is comprised of a math problem, a verified reasoning history for solving the problem, and a subsequent reasoning step that needs to be verified. 

```
Question: "A math problem."

Reasoning history: "A verified reasoning history."

Subsequent step: "To-be-verified subsequent reasoning step."
```

## Verification Procedure

Please verify the subsequent step with the assistance of Python programming.

- You must alternately use human and programming languages in the chain of thought, unless the to-be-verified step is too straightforward or unsuitable for program-aided verification.
- You must carefully follow the rationale of the reasoning step, and verify each operation and logic for correctness.
- Conclude the correctness of the to-be-verified step when observations are sufficient and encapsulate the verification result with Markdown bold syntax (**RIGHT** / **WRONG**), and end your conclusion with the special token "[END]" to denote the completion of your response. 

### Programming Guidelines

Consider using Sympy or Numpy library to facilitate your derivation, calculation and equation solving. Utilize the "pi" symbol and "Rational" from Sympy for $$\pi$$ and fractions, and simplify all fractions and square roots without converting them to decimal values. Please encapsulate each generated Jupyter Python code block with tags "<python>" and "</python>".

### Verification Dimensions and Guidelines

To verify more accurately and comprehensively, focus on the following dimensions of the to-be-verified step.

#### Operation Accuracy

The accuracy of numerical and algebraic operations.

- Whether the equivalence relation between both sides of an equation is established, especially in numerical calculations and algebraic expression simplifications.
- Whether the solution to an equation is correct.
- Whether the references to the values and expressions in the preceding problem and reasoning history are precise. 

#### Logic Soundness

The logic soundness of analysis and deduction.

- Whether the logic behind each analysis and deduction is valid.
- Whether the logical relationship between each step within the to-be-verified step hold.
- Whether the logical relationship between the to-be-verified step and the preceding problem and reasoning history hold."""

ci_ver_usr = "### Problem\n\n{problem}\n\n### Reasoning history\n\n{reasoning_history}\n\n### Current step\n\n{current_step}"


class Steps2Ver(BaseModel):
    problem: str = Field(description="A math problem")
    steps: List[str] = Field(description='A series of reasoning step for problem-solving')

class CI_Verifier(Agent):

    def __init__(
        self,
        llm: Union[BaseLLM, Dict],
        usr_templates: List[str],
        finish_condition: Callable[
            [AgentMessage],
            bool] = lambda m: m.formatted['status'] in ["**WRONG**", "Invalid response: No keywords found"]
    ):
        super().__init__()
        self.completeness_checker = Agent(
            llm = gpt_api,
            template = complete_check_sys,
            output_format = dict(
                type = KeyParser,
                keywords = ["**COMPLETE**", "**INCOMPLETE**"],
            ),
        )
        self.nl_verifier = Agent(
            llm = gpt_api,
            template = nl_ver_sys,
            output_format = dict(
                type = KeyParser,
                keywords = ["**RIGHT**", "**WRONG**", "**NEED FURTHER VERIFICATION**"],
            ),
        )
        self.ci_verifier = MathCoder(
            llm=llm,
            output_format=ToolParser(
                tool_type='interpreter',
                template=ci_ver_sys,
                begin='<python>\n',
                end='\n</python>'),
            aggregator=InternLMToolAggregator(
                environment_role='system',
                environment_begin='<output>\n',
                environment_end='\n</output>'),
            finish_condition=lambda m: '[END]' in m.content,
        )
        self.usr_templates = usr_templates
        self.finish_condition = finish_condition

    # get unique id for each step
    def _get_unique_id(self, data_idx, step_idx):
        return data_idx * 100 + step_idx

    def __call__(self, steps2ver: Steps2Ver, data_idx: int, **kwargs):
        problem = steps2ver.problem
        steps = steps2ver.steps

        # TODO: 优化一下可视化方案
        for step_idx in range(len(steps)):
            session_id = self._get_unique_id(data_idx, step_idx)

            print(session_id)

            if step_idx == 0:
                current_step_ids = []
            current_step_ids.append(step_idx)
            previous_steps = "\n\n".join(steps[:current_step_ids[0]]) if current_step_ids[0] else "No previous steps as the current reasoning step is the first one"
            current_step = "\n\n".join([steps[i] for i in current_step_ids])
            following_steps = "\n\n".join(steps[step_idx + 1:]) if step_idx + 1 < len(steps) else "No following steps as the current reasoning step is the last one"
            
            # stage 1: completeness check TODO: 后续处理一下当前步骤为最后一步的corner case
            message = self.usr_templates[0].format(
                problem = problem,
                previous_steps = previous_steps,
                current_step = current_step,
                following_steps = following_steps,
            )
            message = self.completeness_checker(AgentMessage(sender = "user", content = message), session_id = session_id)
            # self.update_memory(self.completeness_checker.memory.get_memory(session_id), session_id)
            if self.finish_condition(message):
                break
            elif message.formatted["status"] == "**INCOMPLETE**":
                continue
            else:
                current_step_ids = []
            
            # stage 2: NL-verification
            message = self.usr_templates[1].format(
                problem = problem,
                reasoning_history = previous_steps,
                current_step = current_step,
            )
            message = self.nl_verifier(AgentMessage(sender = "user", content = message), session_id = session_id)
            # self.update_memory(self.nl_verifier.memory.get_memory(session_id), session_id)

            if self.finish_condition(message):
                break
            elif message.formatted["status"] == "**RIGHT**" :
                continue
            
            # stage 3: CI-verification
            message = self.usr_templates[2].format(
                problem = problem,
                reasoning_history = previous_steps,
                current_step = current_step,
            )
            message = self.ci_verifier(AgentMessage(sender = "user", content = message), session_id = session_id)
            # self.update_memory(self.ci_verifier.memory.get_memory(session_id), session_id)
            if self.finish_condition(message):
                break
        stop_idx = step_idx
        return self._gather(steps2ver, data_idx, stop_idx)

    # 返回完整的验证过程
    def _gather(self, steps2ver: Steps2Ver, data_idx: int, stop_idx: int):
        data_dict = dict(
            data_idx = data_idx,
            problem = steps2ver.problem,
            steps = steps2ver.steps,
            step_res = [],
            verification = [],
        )
        for step_idx in range(stop_idx + 1):
            step_ver_process = []
            session_id = self._get_unique_id(data_idx, step_idx)
            messages = self.completeness_checker.memory.get_memory(session_id)
            for msg in messages:
                step_ver_process.append(
                    dict(
                        role = "user" if msg.sender == "user" else "language",
                        content = msg.content,
                    )
                )
            step_res = msg.formatted["status"]
            
            if session_id in self.nl_verifier.memory.memory_map:
                messages = self.nl_verifier.memory.get_memory(session_id)
                for msg in messages:
                    step_ver_process.append(
                        dict(
                            role = "user" if msg.sender == "user" else "language",
                            content = msg.content,
                        )
                    )
                step_res = msg.formatted["status"]

            if session_id in self.ci_verifier.memory.memory_map:
                step_ver_process.append(self.ci_verifier.get_steps(session_id))
                step_res = "**RIGHT**" if "**RIGHT**" in self.ci_verifier.get_steps(session_id)[-1]["content"] else "**WRONG**"

            data_dict["step_res"].append(step_res)
            data_dict["verification"].append(step_ver_process)


        return data_dict




        
    # # 返回完整的验证过程
    # def get_steps(self, session_id):
    #     steps, tool_type = [], None
    #     for msg in self.agent.memory.get_memory(session_id):
    #         if msg.formatted:
    #             steps.append(
    #                 dict(role='language', content=msg.formatted['thought']))
    #             if msg.formatted['tool_type']:
    #                 tool_type = msg.formatted['tool_type']
    #                 steps.append(
    #                     dict(
    #                         role='tool',
    #                         content=msg.formatted['action'],
    #                         name=tool_type))
    #         elif msg.sender != 'user':
    #             feedback = dict(role='environment', content=msg.content)
    #             if tool_type:
    #                 feedback['name'] = tool_type
    #             steps.append(feedback)
    #     return steps




if __name__ == "__main__":

    import json
    from lagent.llms import GPTAPI

    gpt_api = GPTAPI(
        model_type = 'gpt-4o-2024-08-06',
        key = "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", # TODO: add api key
        max_new_tokens=4096,
        proxies=dict(
            http=
            'http://konglingkai:T4mEirDBCCcVkcFfA1mEwDnx7hz3gyHgt9clhVO1i8YmMXFIuPTaTSfBUfU0@closeai-proxy.pjlab.org.cn:23128',
            https=
            'http://konglingkai:T4mEirDBCCcVkcFfA1mEwDnx7hz3gyHgt9clhVO1i8YmMXFIuPTaTSfBUfU0@closeai-proxy.pjlab.org.cn:23128',
        ),
        retry=1000,
    )

    ci_verifier = CI_Verifier(
        llm = gpt_api,
        usr_templates = [
            complete_check_usr,
            nl_ver_usr,
            ci_ver_usr,
        ]
    )

    problem = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"
    steps = [
        "The problem asks us to convert the point $(0,3)$ in rectangular coordinates to polar coordinates. We need to express the answer in the form $(r, \\theta)$, where $r > 0$ and $0 \\leq \\theta < 2\\pi$.",
        "By definition, the polar coordinates of a point $(x, y)$ in rectangular coordinates are given by $(r, \\theta)$, where $r$ is the distance from the origin to the point and $\\theta$ is the angle between the positive x-axis and the line connecting the origin to the point.",
        "In this case, the point $(0, 3)$ lies on the y-axis, which means the angle $\\theta$ is $\\frac{\\pi}{2}$.",
        "The distance $r$ from the origin to the point $(0, 3)$ is given by the length of the line segment connecting the origin to the point, which is the y-coordinate of the point. Therefore, $r = 3$.",
        "Hence, the polar coordinates of the point $(0, 3)$ are $\\boxed{(3, \\frac{\\pi}{2})}$.  #### (3, \\frac{\\pi}{2}).",
    ]

    ver_process = ci_verifier(Steps2Ver(problem = problem, steps = steps), data_idx = 0)
    print(ver_process)
    with open("/cpfs01/user/konglingkai/lagent/lagent/agents/work_dir/test.json", "w", encoding = "utf-8") as f:
        json.dump(ver_process, f, ensure_ascii=False, indent=4)




