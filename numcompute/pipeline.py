import numpy as np
class Transformer:
    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)
    

    
class Estimator:
    def fit(self, X, y):
        return self

    def predict(self, X):
        raise NotImplementedError(
            "predict() must be implemented by subclasses."
            "The base Estimator class does not make predictions."
        )
    


class Pipeline:
    def __init__(self, steps):
        if not steps:
            raise ValueError("Pipeline must have at least one step.")
        
        for i, step in enumerate(steps):
            if not isinstance(step, (tuple, list) or len(step) != 2):
                raise ValueError(
                    f"Step {i} is not valid. Each step must be a tuple of "
                    f"(name, transformer), e.g. ('scale', StandardScaler()). "
                    f"Got: {step}"
                )
            name, obj = step

            if not isinstance(name, str):
                raise ValueError(
                    f"Step {i}: the name must be a string. Got: {type(name)}"
                )

            if "__" in name:
                raise ValueError(
                    f"Step name '{name}' cannot contain '__' (double underscore). "
                    f"Please choose a simpler name."
                )
        
        for i, (name, obj) in enumerate(steps[:-1]):
            if not hasattr(obj, "transform"):
                raise ValueError(
                    f"Step '{name}' (position {i}) must have a transform() method. "
                    f"Only the final step is allowed to be an Estimator without transform()."
                )


        self.steps = steps

    def _get_transformers(self):
        return self.steps[:-1]

    def _get_final_step(self):
        return self.steps[-1]

    def fit(self, X, y=None):
        X_current = X   # start with the original data
        for name, transformer in self._get_transformers():
            X_current = transformer.fit_transform(X_current)
        final_name, final_step = self._get_final_step()

        if isinstance(final_step, Estimator):
            final_step.fit(X_current, y)
        else:
            final_step.fit(X_current)
        return self

    def transform(self, X):
        final_name, final_step = self._get_final_step()
        if not hasattr(final_step, "transform"):
            raise TypeError(
                f"The final step '{final_name}' does not have a transform() method. "
                f"If the final step is a model, use predict() instead of transform()."
            )

        X_current = X
        for name, transformer in self.steps:
            X_current = transformer.transform(X_current)
        return X_current

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def predict(self, X):
        final_name, final_step = self._get_final_step()
        if not hasattr(final_step, "predict"):
            raise TypeError(
                f"The final step '{final_name}' does not have a predict() method. "
                f"Use transform() instead if the final step is a Transformer."
            )
        X_current = X
        for name, transformer in self._get_transformers():
            X_current = transformer.transform(X_current)
        return final_step.predict(X_current)

    def get_step(self, name):
        for step_name, obj in self.steps:
            if step_name == name:
                return obj
        raise KeyError(
            f"No step named '{name}' found in the pipeline. "
            f"Available step names: {[s[0] for s in self.steps]}"
        )

    def __repr__(self):
        steps_str = ", ".join(
            f"('{name}', {obj.__class__.__name__}())"
            for name, obj in self.steps
        )
        return f"Pipeline([{steps_str}])"
