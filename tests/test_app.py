from app.app import FEATURE_COLUMNS, create_app


class StubModel:
    def __init__(self):
        self.inputs = None

    def predict(self, inputs):
        self.inputs = inputs
        return [42.5]


def valid_form_data():
    return {
        "cement": "350",
        "blast_furnace_slag": "0",
        "fly_ash": "0",
        "water": "180",
        "superplasticizer": "8",
        "coarse_aggregate": "1000",
        "fine_aggregate": "750",
        "age": "28",
    }


def test_home_page_displays_all_model_inputs():
    client = create_app(model=StubModel()).test_client()

    response = client.get("/")

    assert response.status_code == 200
    for feature in FEATURE_COLUMNS:
        assert feature.encode() in response.data


def test_valid_form_predicts_using_model_feature_order():
    model = StubModel()
    client = create_app(model=model).test_client()

    response = client.post("/", data=valid_form_data())

    assert response.status_code == 200
    assert b"42.50 MPa" in response.data
    assert list(model.inputs.columns) == list(FEATURE_COLUMNS)
    assert model.inputs.iloc[0].tolist() == [350, 0, 0, 180, 8, 1000, 750, 28]


def test_invalid_values_are_reported_without_predicting():
    model = StubModel()
    client = create_app(model=model).test_client()
    form_data = valid_form_data()
    form_data["cement"] = "-1"
    form_data["age"] = "0"

    response = client.post("/", data=form_data)

    assert response.status_code == 400
    assert b"Value cannot be negative." in response.data
    assert b"Curing age must be greater than zero." in response.data
    assert model.inputs is None


def test_non_numeric_values_are_reported_without_predicting():
    model = StubModel()
    client = create_app(model=model).test_client()
    form_data = valid_form_data()
    form_data["water"] = "not a number"

    response = client.post("/", data=form_data)

    assert response.status_code == 400
    assert b"Enter a valid number." in response.data
    assert model.inputs is None
